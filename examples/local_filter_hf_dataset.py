"""
This file contains code to:
1 - Load a parquet-format Hugging Face dataset from the hub.
2 - Filter the dataset (to include only entries that contain the word 'hugging' in the text column).
3 - Push the filtered dataset back to the hub.
"""

import argparse


parser = argparse.ArgumentParser("Filter an HF dataset and push the result to the hub")

parser.add_argument("input_dataset", type=str, help="HF dataset to filter")
parser.add_argument("--n_tasks", type=int, help="number of tasks", default=100)
parser.add_argument("--text_key", type=str, help="text column", default="function")

ORG_NAME = "JoyboyBrian"
LOCAL_PATH = "/home/ubuntu/datatrove/test-local-path"
LOCAL_LOGS_PATH = "/home/ubuntu/datatrove/my_local_logs_path"

if __name__ == "__main__":
    args = parser.parse_args()
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.filters import LambdaFilter
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter
    
    dist_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(args.input_dataset, glob_pattern="*.parquet", text_key=args.text_key),
            LambdaFilter(lambda doc: "query_with_pdf" in doc.text),  # add your custom filter here
            JsonlWriter(
                output_folder="LOCAL_PATH",
            ),
        ],
        tasks=args.n_tasks,
        logging_dir=f"{LOCAL_LOGS_PATH}/",
    )
    dist_executor.run()
