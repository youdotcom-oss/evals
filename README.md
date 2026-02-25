# `evals`: An Evaluation Framework for Search APIs

This repository contains evaluation framework for AI-first web search APIs. Each API is integrated as a sampler and evaluated across benchmarks that test accuracy, latency, and information retrieval performance.

The framework supports multiple search providers (You.com, Exa, Perplexity, Tavily, Parallel) and a representative Google SERP–based sampler. For each query, search results are fetched from the search API, synthesized into an answer using an LLM, then graded against the ground truth.[^1] Benchmarks include [SimpleQA](https://openai.com/index/introducing-simpleqa/) (factual question answering) and [FRAMES](https://arxiv.org/abs/2409.12941) (deep research and multi-hop reasoning). Additional samplers and datasets can be integrated via the configs (see `src/evals/configs/`).


To learn more about our evals methodology and system architecture, please read You.com's research articles:
- [How to Evaluate AI Search in the Agentic Era: A Sneak Peek](https://you.com/resources/sneak-peek-how-to-evaluate-ai-search-in-the-agentic-era)
- [How to Evaluate AI Search for the Agentic Era](https://you.com/resources/how-we-evaluate-ai-search)
- [Randomness in AI Benchmarks: What Makes an Eval Trustworthy?](https://you.com/resources/randomness-in-ai-benchmarks)

## Results

Below are evaluation results across different search samplers and benchmark suites. Grading is performed via an LLM judge using prompts from the standard benchmarks (as specified in the original papers or repositories).[^2]

**SimpleQA**

| sampler                   | accuracy | avg_latency_ms |
|---------------------------|----------|----------------|
| exa_search_with_text      | 91.79%   | 1403.72        |
| google_serp               | 83.01%   | 2050.33        |
| parallel_search_one_shot  | 92.05%   | 3549.39        |
| perplexity_search         | 93.76%   | **339.38**         |
| tavily_advanced           | 91.66%   | 2677.22        |
| tavily_basic              | 61.42%   | 1499.55        |
| you_search                | 85.69%   | 711.59         |
| you_search_with_livecrawl | **94.15%**   | 1158.12        |


**FRAMES**

| sampler                   | accuracy | avg_latency_ms |
|---------------------------|----------|----------------|
| exa_search_with_text      | 47.82%   | 1464.63        |
| google_serp               | 36.65%   | 2211.16        |
| parallel_search_one_shot  | 48.06%   | 3635.09        |
| perplexity_search         | 46%      | **508.77**         |
| tavily_advanced           | 50.55%   | 2690.5         |
| tavily_basic              | 32.35%   | 2201.62        |
| you_search                | 40.17%   | 620.2          |
| you_search_with_livecrawl | **68.93%**   | 938.25         |



## Installation

```bash
# Clone the repository
git clone https://github.com/youdotcom-oss/evals.git
cd evals

# Create a virtual environment, then install
pip install -r requirements.txt
pip install -e .
```

### API keys

Copy the example env file and set the appropriate API keys for the samplers you want to run:

```bash
cp .env.example .env
```

Edit `.env` and set the keys for your chosen providers. To run evaluations for a given search API, set the corresponding environment variable to a valid API key, then pass the sampler name via `--samplers`:

| Sampler                     | Environment variable   |
|-----------------------------|-------------------------|
| Exa                         | `EXA_API_KEY`           |
| Google                      | `SERP_API_KEY`           |
| Parallel                    | `PARALLEL_API_KEY`      |
| Perplexity                  | `PERPLEXITY_API_KEY`    |
| Tavily (basic / advanced)   | `TAVILY_API_KEY`        |
| You.com                     | `YOU_API_KEY`           |

Grading uses an OpenAI model; set `OPENAI_API_KEY` for the LLM judge.

## Usage

### Basic instructions

Run evaluations from the command line via the eval runner:

```bash
# List available samplers and datasets
python src/evals/eval_runner.py --help

# Run SimpleQA on all samplers (default)
python src/evals/eval_runner.py

# Run SimpleQA for specific samplers only
python src/evals/eval_runner.py --samplers you_search_with_livecrawl tavily_basic --datasets simpleqa

# Run FRAMES evaluation
python src/evals/eval_runner.py --datasets frames

# Run on a limited number of problems (e.g. 100 for a quick sanity check)
python src/evals/eval_runner.py --samplers you_search_with_livecrawl --datasets simpleqa --limit 100

# Fresh run: clear existing results and re-run
python src/evals/eval_runner.py --clean --samplers you_search_with_livecrawl --datasets simpleqa --limit 100
```

### Benchmark suites

| Benchmark | Description | Flag / usage |
|-----------|-------------|--------------|
| SimpleQA | Factual question answering ([OpenAI SimpleQA](https://openai.com/index/introducing-simpleqa/)) | `--datasets simpleqa` |
| FRAMES   | Deep research and multi-hop reasoning ([paper](https://arxiv.org/abs/2409.12941), [dataset](https://huggingface.co/datasets/google/frames-benchmark)) | `--datasets frames` |

### Other configuration options

| Option              | Flag / default        | Description |
|---------------------|------------------------|-------------|
| Samplers            | `--samplers <names>`   | One or more sampler names (default: all). |
| Datasets            | `--datasets <names>`   | One or more datasets: `simpleqa`, `frames` (default: all). |
| Limit               | `--limit <n>`          | Run on at most `n` problems (optional). |
| Batch size          | `--batch-size 50`      | Number of problems per batch before writing results (default: 50). |
| Max concurrent tasks| `--max-concurrent-tasks 10` | Concurrency limit (default: 10). |
| Clean               | `--clean`              | Remove existing results and run from scratch. |

## Output

Results are written to `src/evals/results/` with the following structure:

```
src/evals/results/
├── dataset_simpleqa_raw_results_<sampler_name>.csv   # Per-sampler, per-dataset raw results
├── dataset_frames_raw_results_<sampler_name>.csv
└── analyzed_results.csv                               # Aggregated metrics (accuracy, latency)
```

Raw CSVs contain per-query fields (e.g. query, generated answer, evaluation result, latencies). After a run, `write_metrics()` is called automatically and `analyzed_results.csv` is updated with accuracy and average latency per sampler and dataset.

## Citation

If you use this repository in your research, please consider citing:

```bibtex
@misc{2026yousearchevals,
  title        = {evals: An Evaluation Framework for AI-first Web Search APIs},
  author       = {You.com},
  year         = {2026},
  journal      = {GitHub repository},
	publisher    = {GitHub},
  howpublished = {\url{https://github.com/youdotcom-oss/evals}}
}
```

## License

This repository is made available under the [MIT License](LICENSE).

## Notes

[^1]: Search results are fetched from each search API, then synthesized into a single answer using an LLM; the answer is graded by an LLM judge. Synthesis uses gpt-4o-mini and grading uses gpt-4.1 (configurable in `src/evals/constants.py`).
[^2]: Grading uses prompts aligned with the standard benchmarks as specified in the original papers or repositories (e.g. [SimpleQA](https://openai.com/index/introducing-simpleqa/) and [FRAMES](https://arxiv.org/abs/2409.12941).
