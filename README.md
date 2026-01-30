# API Evaluations
This repository provides a framework for running evaluations, including [OpenAI's SimpleQA evaluation](https://openai.com/index/introducing-simpleqa/). 
This code was used to evaluate the APIs in [this You.com blogpost](https://home.you.com/articles/search-api-for-the-agentic-era).

If you would like to reproduce the numbers or add new samplers, follow the instructions on how to install and run the code.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/youdotcom-oss/evals.git
   cd evals
   ```

2. Create a virtual environment with the tool of your choice, then install the required dependencies:
   ```bash
   # create and activate virtual environment
   pip install -r requirements.txt
   pip install -e .
   ```

3. Set up your `.env` file and insert the appropriate API keys:
   ```bash
   cp .env.example .env
   ```

## Running a SimpleQA evaluation
To run a SimpleQA evaluation, simply run the `simpleqa_runner.py` file with your desired arguments.

View available arguments and samplers
   ```bash
   python src/evals/eval_runner.py --help
   ```

Run the SimpleQA evaluation on the entire problem set for all available samplers with default settings
   ```bash
   python src/evals/eval_runner.py
   ```

Run the SimpleQA evaluation on just You.com for 5 random problems
   ```bash
   python src/evals/eval_runner.py --samplers you_unified_search --limit 5
   ```

## Interpreting Results 
Results files will be placed in `simpleqa/results` after a successful run of SimpleQA. Files following the pattern
`raw_results_{sampler}.csv` are the raw results for each individual sampler. The file `simpleqa_results.csv` contains
aggregated results with various metrics useful for analysis.