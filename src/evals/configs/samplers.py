import os

from evals.samplers.applied_samplers.exa_sampler import ExaSampler
from evals.samplers.applied_samplers.serp_api_google_sampler import SerpApiGoogleSampler
from evals.samplers.applied_samplers.tavily_sampler import TavilySampler
from evals.samplers.applied_samplers.you_sampler import YouSampler

SAMPLERS = [
    YouSampler(
        sampler_name="you_unified_search",
        api_key=os.getenv("YOU_API_KEY"),
    ),
    ExaSampler(
        sampler_name="exa_search_with_contents",
        api_key=os.getenv("EXA_API_KEY"),
        custom_args={"text": True},
    ),
    SerpApiGoogleSampler(
        sampler_name="serp_google",
        api_key=os.getenv("SERP_API_KEY"),
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