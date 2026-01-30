"""
Analyze and calculate metrics from evaluation results.

This module provides functionality to process raw evaluation results,
calculate performance metrics, and generate summary reports.
"""

import glob
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


def get_default_results_dir() -> Path:
    """Get the default results directory path."""
    return Path(os.getcwd(), "src/evals/results")


def write_metrics(results_dir: Optional[Path] = None):
    """
    Calculate metrics from raw results such as accuracy score, P50 latency, and average latency.

    Args:
        results_dir: Optional path to results directory. Defaults to src/evals/results
    """
    if results_dir is None:
        results_dir = get_default_results_dir()

    files = glob.glob(f"{results_dir}/raw_results_*.csv")
    metric_rows = []

    for sampler_results_file in files:
        sampler_name = sampler_results_file.split("raw_results_")[-1].split(".")[0]
        df_sampler_results = pd.read_csv(sampler_results_file)
        successful_df = df_sampler_results[df_sampler_results["response_time_ms"] != "FAILED"]

        p50_latency = pd.to_numeric(successful_df["response_time_ms"]).median()
        avg_latency = pd.to_numeric(successful_df["response_time_ms"]).mean()
        correct = len(df_sampler_results[df_sampler_results["evaluation_result"] == "is_correct"])
        count_answered = len(successful_df)

        if count_answered == 0:
            raise ValueError(f"No successful results found for sampler {sampler_name}")

        accuracy_score = round((correct / count_answered) * 100, 2)

        metric_rows.append({
            "provider": sampler_name,
            "accuracy_score": accuracy_score,
            "p50_latency": round(float(p50_latency), 2),
            "avg_latency": round(float(avg_latency), 2),
            "problem_count": count_answered,
        })

    write_path = results_dir / "simpleqa_results.csv"
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(write_path, index=False)
    print(f"Results were written to {write_path}")
    print(metric_df)


def calculate_sampler_metrics(sampler_results_file: str) -> Dict[str, Any]:
    """
    Calculate metrics for a single sampler's results.

    Args:
        sampler_results_file: Path to the raw results CSV file

    Returns:
        Dictionary containing calculated metrics
    """
    sampler_name = sampler_results_file.split("raw_results_")[-1].split(".")[0]
    df_sampler_results = pd.read_csv(sampler_results_file)
    successful_df = df_sampler_results[df_sampler_results["response_time_ms"] != "FAILED"]

    p50_latency = pd.to_numeric(successful_df["response_time_ms"]).median()
    avg_latency = pd.to_numeric(successful_df["response_time_ms"]).mean()
    correct = len(df_sampler_results[df_sampler_results["evaluation_result"] == "is_correct"])
    count_answered = len(successful_df)

    if count_answered == 0:
        raise ValueError(f"No successful results found for sampler {sampler_name}")

    accuracy_score = round((correct / count_answered) * 100, 2)

    return {
        "provider": sampler_name,
        "accuracy_score": accuracy_score,
        "p50_latency": round(float(p50_latency), 2),
        "avg_latency": round(float(avg_latency), 2),
        "problem_count": count_answered,
    }


def get_results_files(results_dir: Optional[Path] = None) -> List[str]:
    """
    Get all raw results files from the results directory.

    Args:
        results_dir: Optional path to results directory. Defaults to src/evals/results

    Returns:
        List of file paths to raw results files
    """
    if results_dir is None:
        results_dir = get_default_results_dir()

    return glob.glob(f"{results_dir}/raw_results_*.csv")
