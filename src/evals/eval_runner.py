"""
The main file for running evals. Use this file to run the eval against your selected samplers and datasets.
Available samplers can be found using --help or in configs/samplers
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd
from tqdm import tqdm

from evals.configs import samplers, datasets
from evals.samplers.base_samplers.base_sampler import BaseSampler
from evals.eval_results_analyzer import write_metrics, get_default_results_dir


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# Mute noisy client logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def get_sampler_filepath(sampler: BaseSampler, dataset: datasets.Dataset, results_dir: Path = None) -> Path:
    """Get the filepath for a sampler's results file."""
    if results_dir is None:
        results_dir = get_default_results_dir()

    return results_dir / f"dataset_{dataset.dataset_name}_raw_results_{sampler.sampler_name}.csv"


def get_sampler(sampler_name: str):
    """Initialize requested samplers"""
    sampler = next((sampler for sampler in samplers.SAMPLERS if sampler.sampler_name == sampler_name), None)
    if sampler is None:
        raise ValueError(
            f"Sampler '{sampler_name}' not found. Available samplers: {[sampler.sampler_name for sampler in samplers.SAMPLERS]}"
        )
    return sampler


def clean_results_folder(results_dir: Path = None):
    """Clean the results folder."""
    if results_dir is None:
        results_dir = get_default_results_dir()
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)


def get_remaining_problems(dataset: datasets.Dataset, sampler: BaseSampler, results_dir: Path = None):
    """In case of failure, only run problems from the dataset that have not been run yet"""
    if results_dir is None:
        results_dir = get_default_results_dir()
    sampler_results_filepath = get_sampler_filepath(sampler, dataset, results_dir)
    if os.path.isdir(results_dir) and os.path.isfile(sampler_results_filepath):
        sampler_results = pd.read_csv(sampler_results_filepath)
        return dataset.df[~dataset.df["problem"].isin(sampler_results["query"].tolist())]
    return dataset.df


async def process_query_with_semaphore(semaphore, sampler, target_query, target_ground_truth, dataset):
    async with semaphore:
        try:
            return await sampler(target_query, ground_truth=target_ground_truth, dataset=dataset)
        except Exception as e:
            logging.error(f"Failed to run {sampler.sampler_name} for query: {target_query}")
            raise e


def get_dataset(dataset_name):
    dataset = next((dataset for dataset in datasets.DATASETS if dataset.dataset_name == dataset_name), None)
    dataset.df = pd.read_csv(dataset.csv_path)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not recognized, run python src/evals/eval_runner.py --help for available datasets")
    if dataset.df is None:
        raise ValueError(f"Failed to initialize df for {dataset_name} and csv_path {dataset.csv_path}")
    return dataset


async def run_evals(
    args: argparse.Namespace,
    results_dir: Path = None,
):
    """
    Run benchmark for each sampler.

    Run the selected number of queries against each requested sampler. Creates all tasks upfront with concurrency
    controlled by semaphore, and provides a progress bar to track progress throughout the run.

    Args:
        args: Command line arguments
        results_dir: Directory to write results to. Defaults to src/evals/results
    """
    if results_dir is None:
        results_dir = get_default_results_dir()

    if args.clean:
        clean_results_folder(results_dir)

    results = {}
    for dataset_name in args.datasets:
        dataset = get_dataset(dataset_name)
        if args.limit:
            dataset.df = dataset.df.sample(n=args.limit)
        for sampler_name in args.samplers:
            sampler = get_sampler(sampler_name)
            # Only run on problems that are not already in results folder
            remaining_problems = get_remaining_problems(dataset=dataset, sampler=sampler, results_dir=results_dir)
            if len(remaining_problems) == 0:
                logging.info(f"No problems remaining for sampler {sampler.sampler_name}, moving on...")
                results[sampler.sampler_name] = pd.read_csv(get_sampler_filepath(sampler, dataset, results_dir))
                continue

            logging.info(f"Running sampler {sampler.sampler_name} on dataset {dataset_name} on {len(remaining_problems)} problems")
            dataset.df = remaining_problems

            with tqdm(
                total=len(dataset.df),
                desc=f"Running sampler: {sampler.sampler_name} for dataset {dataset.dataset_name}",
                unit="queries",
            ) as pbar:
                semaphore = asyncio.Semaphore(args.max_concurrent_tasks)
                tasks = []
                # Create tasks all at once
                for _, row in dataset.df.iterrows():
                    query = row["problem"]
                    ground_truth = row["answer"]
                    task = asyncio.create_task(
                        process_query_with_semaphore(
                            semaphore=semaphore,
                            sampler=sampler,
                            target_query=query,
                            target_ground_truth=ground_truth,
                            dataset=dataset,
                        )
                    )
                    tasks.append(task)

                batch_results = []
                # Return results as completed
                for coroutine in asyncio.as_completed(tasks):
                    result = await coroutine
                    batch_results.append(result)
                    pbar.update(1)

                    if len(batch_results) >= args.batch_size:
                        write_raw_sampler_results(batch_results, sampler, dataset, results_dir)
                        batch_results = []

                if batch_results:
                    write_raw_sampler_results(batch_results, sampler, dataset, results_dir)


def write_raw_sampler_results(sampler_results: list[str | Any], sampler: BaseSampler, dataset: datasets.Dataset, results_dir: Path = None):
    """
    Write raw results to a csv file.

    This takes the raw results list, not the full results dictionary in case an individual sampler fails.
    """
    if results_dir is None:
        results_dir = get_default_results_dir()

    df_sampler_results = pd.DataFrame(sampler_results)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    sampler_results_filepath = get_sampler_filepath(sampler, dataset, results_dir)
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


async def main():
    available_samplers = [sampler.sampler_name for sampler in samplers.SAMPLERS]
    available_datasets = [dataset.dataset_name for dataset in datasets.DATASETS]
    parser = argparse.ArgumentParser(description="Run an eval")
    parser.add_argument(
        "--samplers",
        default=available_samplers,
        type=str,
        nargs="+",
        help=f"List of samplers to run. Choose from {available_samplers}",
    )
    parser.add_argument(
        "--datasets",
        default=available_datasets,
        type=str,
        nargs="+",
        help=f"The dataset(s) to eval against (can specify multiple). Select from {available_datasets}",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Determines the amount of problems to evaluate against",
    )
    parser.add_argument(
        "--batch-size",
        default=50,
        type=int,
        help="Number of results to accumulate before writing to disk (for incremental progress tracking)",
    )
    parser.add_argument(
        "--max-concurrent-tasks",
        default=10,
        type=int,
        help="Maximum number of concurrent async tasks (controls parallelism via semaphore)",
    )
    parser.add_argument(
        "--clean",
        default=False,
        type=str,
        help="If set to True, wipes results folder if it exists to set up a fresh run on all samplers and all problems. If set to False, the evaluation is only run for samplers and problems that do not already exist in results folder",
    )

    args = parser.parse_args()

    await run_evals(args)

    write_metrics()


if __name__ == "__main__":
    asyncio.run(main())
