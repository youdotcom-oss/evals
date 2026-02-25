import pandas as pd

from evals.configs import datasets, samplers


def get_dataset(dataset_name):
    dataset = next(
        (
            dataset
            for dataset in datasets.DATASETS
            if dataset.dataset_name == dataset_name
        ),
        None,
    )
    if dataset is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not recognized, run python src/evals/eval_runner.py --help for available datasets"
        )
    dataset.df = pd.read_csv(dataset.csv_path)
    if dataset.df is None:
        raise ValueError(
            f"Failed to initialize df for {dataset_name} and csv_path {dataset.csv_path}"
        )
    return dataset


def get_sampler(sampler_name: str):
    sampler = next(
        (
            sampler
            for sampler in samplers.SAMPLERS
            if sampler.sampler_name == sampler_name
        ),
        None,
    )
    if sampler is None:
        raise ValueError(
            f"Sampler '{sampler_name}' not found. Available samplers: {[sampler.sampler_name for sampler in samplers.SAMPLERS]}"
        )
    return sampler
