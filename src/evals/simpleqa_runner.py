"""
The main file for running the SimpleQA eval. Use this file to run the SimpleQA eval against your selected samplers.
Available samplers can be found in get_samplers() or in the `sampler/` folder
"""

import argparse
import asyncio
import glob
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

from evals.configs import samplers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_sampler_filepath(sampler_name):
    return Path(os.getcwd(), f"src/evals/results/raw_results_{sampler_name}.csv")


def get_sampler(sampler_name: str):
    """Initialize requested samplers"""
    sampler = next(
        (sampler for sampler in samplers.SAMPLERS if sampler.sampler_name == sampler_name), None
    )
    if sampler is None:
        raise ValueError(f"Sampler '{sampler_name}' not found")
    return sampler


def clean_results_folder():
    results_folder_path = Path(os.getcwd(), "src/evals/results")
    if os.path.isdir(results_folder_path):
        shutil.rmtree(results_folder_path)


def get_remaining_problems(df, sampler_name):
    """In case of failure, only run problems from the dataset that have not been run yet"""
    sampler_results_filepath = get_sampler_filepath(sampler_name)
    results_folder_path = Path(os.getcwd(), "src/evals/results")
    if os.path.isdir(results_folder_path) and os.path.isfile(sampler_results_filepath):
        sampler_results = pd.read_csv(sampler_results_filepath)
        return df[~df["problem"].isin(sampler_results["query"].tolist())]
    return df


async def process_query_with_semaphore(
    semaphore, sampler, target_query, target_ground_truth
):
    async with semaphore:
        try:
            return await sampler(target_query, ground_truth=target_ground_truth)
        except Exception as e:
            logging.error(
                f"Failed to run {sampler.sampler_name} for query: {target_query}"
            )
            return e


async def get_search_results_and_run_evals(
    args: argparse.Namespace,
):
    """
    Run SimpleQA benchmark for each sampler.

    Run the selected number of SimpleQA queries against each requested sampler. Creates tasks in batches, and provides
    a progress bar to track progress throughout the run. After each sampler is completed, write the results to the
    results folder in the format "raw_results_<sampler>.csv". Once all samplers are completed, calculate metrics based
    on the retrieved results and create a csv called "simpleqa_results.csv".
    """
    df = pd.read_csv(args.csv_path)
    if args.limit:
        df = df.sample(n=args.limit)
    if args.clean:
        clean_results_folder()

    results = {}
    for sampler_name in args.samplers:
        sampler = get_sampler(sampler_name)
        # Only run on problems that are not already in results folder
        remaining_problems = get_remaining_problems(df, sampler.sampler_name)
        if len(remaining_problems) == 0:
            logging.info(
                f"No problems remaining for sampler {sampler.sampler_name}, moving on..."
            )
            results[sampler.sampler_name] = pd.read_csv(
                get_sampler_filepath(sampler.sampler_name)
            )
            continue

        logging.info(
            f"Running sampler {sampler.sampler_name} on {len(remaining_problems)} problems"
        )
        df = remaining_problems

        # Run problems in batches
        with tqdm(
            total=len(df),
            desc=f"Running sampler: {sampler.sampler_name}",
            unit="queries",
        ) as pbar:
            semaphore = asyncio.Semaphore(args.max_concurrent_tasks)

            for i in range(0, len(df), args.batch_size):
                batch_df = df[i : i + args.batch_size]

                tasks = []
                for _, row in batch_df.iterrows():
                    query = row["problem"]
                    ground_truth = row["answer"]
                    task = asyncio.create_task(
                        process_query_with_semaphore(
                            semaphore, sampler, query, ground_truth
                        )
                    )
                    tasks.append(task)

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                pbar.update(len(batch_df))

                await asyncio.gather(*[t for t in tasks if not t.done()])
                # Write results of each batch so we can keep progress in case of a failure
                write_raw_sampler_results(batch_results, sampler.sampler_name)


def write_raw_sampler_results(sampler_results: list[str | Any], sampler_name: str):
    """
    Write raw results to a csv file.

    This takes the raw results list, not the full results dictionary in case an individual sampler fails.
    """
    df_sampler_results = pd.DataFrame(sampler_results)
    if not os.path.isdir(Path(os.getcwd(), "src/evals/results")):
        os.mkdir(Path(os.getcwd(), "src/evals/results"))

    sampler_results_filepath = get_sampler_filepath(sampler_name)
    if os.path.isfile(sampler_results_filepath):
        # If file already exists, append
        df_sampler_results.to_csv(
            sampler_results_filepath,
            index=False,
            header=False,
            mode="a",
        )
    else:
        df_sampler_results.to_csv(
            sampler_results_filepath,
            index=False,
        )


def write_metrics():
    """Calculate metrics from raw results such as average score, P50 latency"""
    results_path = Path(os.getcwd(), "src/evals/results")
    files = glob.glob(f"{results_path}/raw_results_*.csv")
    metric_rows = []
    for sampler_results_file in files:
        sampler_name = sampler_results_file.split("raw_results_")[-1].split(".")[0]
        df_sampler_results = pd.read_csv(sampler_results_file)
        successful_df = df_sampler_results[
            df_sampler_results["response_time_ms"] != "FAILED"
        ]

        # TODO: Mean or median?
        p50_latency = pd.to_numeric(successful_df["response_time_ms"]).median()
        correct = len(
            df_sampler_results[df_sampler_results["evaluation_result"] == "is_correct"]
        )
        count_answered = len(successful_df)
        if count_answered == 0:
            breakpoint()
            raise ValueError("No rows found in raw results file")
        average_score = round((correct / count_answered) * 100, 2)

        metric_rows.append(
            {
                "provider": sampler_name,
                "average_score": average_score,
                "p50_latency": p50_latency,
                "problem_count": count_answered,
            }
        )

    write_path = Path(os.getcwd(), "src/evals/results/simpleqa_results.csv")
    pd.DataFrame(metric_rows).to_csv(write_path, index=False)


async def main():
    available_samplers = ["you_unified_search", "exa", "exa_fast", "google", "tavily"]
    parser = argparse.ArgumentParser(description="Run SimpleQA eval")
    parser.add_argument(
        "--samplers",
        default=available_samplers,
        type=str,
        nargs="+",
        help=f"List of samplers to run. Choose from {available_samplers}",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Determines the amount of problems to evaluate against",
    )
    parser.add_argument(
        "--batch-size",
        default=250,
        type=int,
        help="Used to define the batch size used in multiprocessing. Also determines how many problems will be run before appending to corresponding results file",
    )
    parser.add_argument(
        "--max-concurrent-tasks",
        default=10,
        type=int,
        help="Used to define the max count of concurrent tasks to be used in multiprocessing",
    )
    parser.add_argument(
        "--num-results",
        default=5,
        type=int,
        help="Used to define the number of results returned by each provider",
    )
    parser.add_argument(
        "--csv-path",
        default="src/evals/data/simple_qa_test_set.csv",
        type=str,
        help="Used to define the filepath of the test set",
    )
    parser.add_argument(
        "--clean",
        default=False,
        type=str,
        help="If set to True, wipes results folder if it exists to set up a fresh run on all samplers and all problems. If set to False, the evaluation is only run for samplers and problems that do not already exist in results folder",
    )

    args = parser.parse_args()

    await get_search_results_and_run_evals(args)
    write_metrics()


if __name__ == "__main__":
    asyncio.run(main())
