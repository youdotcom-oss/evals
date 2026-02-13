import os

from evals.samplers.applied_samplers.exa_sampler import ExaSampler
from evals.samplers.applied_samplers.parallel_sampler import ParallelSearchSampler
from evals.samplers.applied_samplers.perplexity_sampler import PerplexitySearchSampler
from evals.samplers.applied_samplers.tavily_sampler import TavilySampler
from evals.samplers.applied_samplers.you_livecrawl_sampler import YouLivecrawlSampler
from evals.samplers.applied_samplers.you_search_sampler import YouSearchSnippetsSampler


SAMPLERS = [
    YouLivecrawlSampler(
        sampler_name="you_search_livecrawl",
        api_key=os.getenv("YOU_API_KEY"),
    ),
    YouSearchSnippetsSampler(
        sampler_name="you_search_snippets",
        api_key=os.getenv("YOU_API_KEY"),
    ),
    ExaSampler(
        sampler_name="exa_search_with_contents",
        api_key=os.getenv("EXA_API_KEY"),
        text=True,
    ),
    ParallelSearchSampler(
        sampler_name="parallel_fast",
        api_key=os.getenv("PARALLEL_API_KEY"),
        mode="fast",
    ),
    PerplexitySearchSampler(
        sampler_name="perplexity_search",
        api_key=os.getenv("PERPLEXITY_API_KEY"),
    ),
    TavilySampler(
        sampler_name="tavily_basic",
        api_key=os.getenv("TAVILY_API_KEY"),
        search_depth="basic",
    ),
    TavilySampler(
        sampler_name="tavily_advanced",
        api_key=os.getenv("TAVILY_API_KEY"),
        search_depth="advanced",
    ),
]
