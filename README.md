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

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Set up environment variables as environment variables or an .env file:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   export YOU_API_KEY=your_you_api_key
   export TAVILY_API_KEY=your_you_api_key
   export EXA_API_KEY=your_you_api_key
   export SERP_API_KEY=your_you_api_key
   ```

## Running a SimpleQA evaluation
To run a SimpleQA evaluation, simply run the `simpleqa_runner.py` file with your desired arguments.

View available arguments and samplers
   ```bash
   python src/simpleqa/simpleqa_runner.py --help
   ```

Run the SimpleQA evaluation on the entire problem set for all available samplers with default settings
   ```bash
   python src/simpleqa/simpleqa_runner.py
   ```

Run the SimpleQA evaluation on just You.com for 5 random problems
   ```bash
   python src/simpleqa/simpleqa_runner.py --samplers you --limit 5
   ```

## Interpreting Results 
Results files will be placed in `simpleqa/results` after a successful run of SimpleQA. Files following the pattern
`raw_results_{sampler}.csv` are the raw results for each individual sampler. The file `simpleqa_results.csv` contains
aggregated results with various metrics useful for analysis.