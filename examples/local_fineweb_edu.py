"""
This file contains the code used to process and create the
FineWeb dataset (https://huggingface.co/datasets/HuggingFaceFW/fineweb)
"""
from multiprocess import freeze_support
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig

if __name__ == '__main__':
    # freeze_support()
    """
    We first run the following pipeline for each dump
    """
    DUMP_TO_PROCESS = "CC-MAIN-2023-50"  # example

    # Adjust paths as needed for your local environment
    MAIN_OUTPUT_PATH = "s3://brian-bucket"  # Ensure your local environment has access to this path or adjust accordingly
    FILTERING_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/base_processing"

    main_processing_executor = LocalPipelineExecutor(
        pipeline=[
            WarcReader(
                f"s3://commoncrawl/crawl-data/{DUMP_TO_PROCESS}/segments/",
                glob_pattern="*/warc/*",  # we want the warc files
                default_metadata={"dump": DUMP_TO_PROCESS},
            ),
            URLFilter(exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/1_url/{DUMP_TO_PROCESS}")),
            Trafilatura(favour_precision=True),
            LanguageFilter(
                exclusion_writer=JsonlWriter(
                    f"{FILTERING_OUTPUT_PATH}/2_non_english/",
                    output_filename="${language}/" + DUMP_TO_PROCESS + "/${rank}.jsonl.gz",
                    # folder structure: language/dump/file
                )
            ),
            GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/3_gopher_rep/{DUMP_TO_PROCESS}")
            ),
            GopherQualityFilter(
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/4_gopher_qual/{DUMP_TO_PROCESS}")
            ),
            C4QualityFilter(
                filter_no_terminal_punct=False,
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/5_c4/{DUMP_TO_PROCESS}"),
            ),
            FineWebQualityFilter(
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/6_fineweb_qual/{DUMP_TO_PROCESS}"),
            ),
            JsonlWriter(f"{FILTERING_OUTPUT_PATH}/output/{DUMP_TO_PROCESS}"),
        ],
        tasks=32,
        workers=32,  # Use -1 for no limit or set to desired number of workers
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/base_processing/{DUMP_TO_PROCESS}",
        randomize_start_duration=180,  # Don't hit the bucket all at once with the list requests
    )

    main_processing_executor.run()

    """
    We then apply minhash deduplication to each individual dump
    """

    # You can also change ngrams or the number of buckets and their size here
    minhash_config = MinhashConfig(
        hash_config=HashConfig(
            hash_fc="sha1",  # Better precision -> fewer false positives (collisions)
            precision=64,
        ),
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5,
    )

    S3_MINHASH_BASE_PATH = f"{MAIN_OUTPUT_PATH}/minhash"

    S3_LOGS_FOLDER = f"{MAIN_OUTPUT_PATH}/logs/minhash"
    LOCAL_LOGS_FOLDER = "logs/minhash"

    TOTAL_TASKS = 1000

    # This is the original data that we want to deduplicate
    INPUT_READER = JsonlReader(
        f"{FILTERING_OUTPUT_PATH}/output/{DUMP_TO_PROCESS}"
    )  # This is the output from the first part

    # Stage 1 computes minhash signatures for each task (each task gets a set of files)
    stage1 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(
                output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/signatures", config=minhash_config
            ),
        ],
        tasks=TOTAL_TASKS,
        workers=32,  # Use -1 for no limit or set to desired number of workers
        logging_dir=f"{S3_LOGS_FOLDER}/signatures",
        randomize_start_duration=180,
        depends=main_processing_executor,  # Only start after the first one completes
    )

    stage1.run()

    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/signatures",
                output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/buckets",
                config=MinhashConfig(hash_config=minhash_config.hash_config),
            ),
        ],
        tasks=minhash_config.num_buckets * 50,  # The code supports parallelizing each bucket. Here we run 50 workers per bucket
        workers=32,  # Use -1 for no limit or set to desired number of workers
        randomize_start_duration=180,
        logging_dir=f"{S3_LOGS_FOLDER}/buckets",
        depends=stage1,
    )

    stage2.run()

    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/buckets",
                output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/remove_ids",
                config=minhash_config,
            ),
        ],
        tasks=1,  # This step runs on a single task
        workers=1,  # Since tasks=1, set workers=1
        logging_dir=f"{S3_LOGS_FOLDER}/clustering",
        depends=stage2,
    )

    stage3.run()

    stage4 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            TokensCounter(),  # You can remove this one; it's just a nice way to know how many tokens we have before and after dedup
            MinhashDedupFilter(input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/remove_ids"),
            # Run the PII removal
            PIIFormatter(),
            JsonlWriter(f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/deduped_output"),
        ],
        tasks=TOTAL_TASKS,
        workers=32,  # Use -1 for no limit or set to desired number of workers
        logging_dir=f"{S3_LOGS_FOLDER}/filtering",
        depends=stage3,
    )

    # Launch dedup pipelines
    stage4.run()
