import os

from evals.samplers.applied_samplers.exa_sampler import ExaSampler
from evals.samplers.applied_samplers.tavily_sampler import TavilySampler
from evals.samplers.applied_samplers.you_livecrawl_sampler import YouLivecrawlSampler
from evals.samplers.applied_samplers.you_search_sampler import YouSearchSampler


SAMPLERS = [
    YouLivecrawlSampler(
        sampler_name="you_search_livecrawl",
        api_key=os.getenv("YOU_API_KEY"),
    ),
    YouSearchSampler(
        sampler_name="you_search",
        api_key=os.getenv("YOU_API_KEY"),
    ),
    ExaSampler(
        sampler_name="exa_search_with_contents",
        api_key=os.getenv("EXA_API_KEY"),
        custom_args={"text": True},
    ),
    TavilySampler(
        sampler_name="tavily_basic",
        api_key=os.getenv("TAVILY_API_KEY"),
        custom_args={"search_depth": "basic"},
    ),
    TavilySampler(
        sampler_name="tavily_advanced",
        api_key=os.getenv("TAVILY_API_KEY"),
        custom_args={"search_depth": "advanced"},
    ),
]
