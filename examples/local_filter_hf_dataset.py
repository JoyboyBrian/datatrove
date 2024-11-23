"""
This file contains code to:
1 - Load a parquet-format Hugging Face dataset from the hub.
2 - Filter the dataset (to include only entries that contain the word 'hugging' in the text column).
3 - Push the filtered dataset back to the hub.
"""

import argparse


parser = argparse.ArgumentParser("Filter an HF dataset and push the result to the hub")

parser.add_argument("input_dataset", type=str, help="HF dataset to filter")
parser.add_argument("output_name", type=str, help="Name of the output dataset")
parser.add_argument("--n_tasks", type=int, help="number of tasks", default=5)
parser.add_argument("--text_key", type=str, help="text column", default="function")

ORG_NAME = "JoyboyBrian"
LOCAL_PATH = "/home/ubuntu/datatrove/test-local-path"
LOCAL_LOGS_PATH = "/home/ubuntu/datatrove/test_local_logs_path"

if __name__ == "__main__":
    args = parser.parse_args()
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.writers.huggingface import HuggingFaceDatasetWriter
    from datatrove.pipeline.filters import LambdaFilter
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter
    from huggingface_hub import create_repo, HfApi

    create_repo(
        repo_id=f"{ORG_NAME}/{args.output_name}",
        repo_type="dataset",
        private=True,
    )
    dist_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(args.input_dataset, glob_pattern="*.parquet", text_key=args.text_key),
            LambdaFilter(lambda doc: "query_with_pdf" in doc.text),  # add your custom filter here
            # JsonlWriter(output_folder="LOCAL_PATH"),
            HuggingFaceDatasetWriter(
                dataset=f"{ORG_NAME}/{args.output_name}",
                private=True,
                local_working_dir=f"{LOCAL_PATH}/{args.output_name}",
                output_filename="data/${rank}.parquet",
                cleanup=True,
            ),
        ],
        tasks=args.n_tasks,
        logging_dir=f"{LOCAL_LOGS_PATH}/",
    )
    dist_executor.run()

# python local_filter_hf_dataset.py hf://datasets/JoyboyBrian/hp_lora_training_data/final_data_nov13 brian-test-local --text_key function