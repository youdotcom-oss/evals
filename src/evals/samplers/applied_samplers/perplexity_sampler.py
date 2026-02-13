import os
from typing import Any

from perplexity import Perplexity

from evals.samplers.base_samplers.base_sdk_sampler import BaseSDKSampler


class PerplexitySearchSampler(BaseSDKSampler):
    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_tokens_per_page: int = None,
        max_tokens: int = 12000,  # Limit max tokens per page to not overload synthesis model
        max_concurrency: int = 10,
    ):
        self.max_tokens_per_page = max_tokens_per_page
        self.max_tokens = max_tokens

        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            timeout=timeout,
            max_concurrency=max_concurrency,
        )

        if api_key is None:
            print("No API key provided for Perplexity")

    def _initialize_client(self):
        """Initialize Perplexity SDK client"""
        self.client = Perplexity(api_key=self.api_key)

    def _get_search_results_impl(self, query):
        return self.client.search.create(
            query=query,
            max_results=10,
            max_tokens_per_page=self.max_tokens_per_page,
            max_tokens=self.max_tokens,
        )

    def format_results(self, results: Any) -> list[str]:
        formatted_results = []
        for result in results.results:
            title = result.title
            url = result.url
            snippet = result.snippet
            # if url not in contaminated_urls:
            formatted_results.append(f"[{title}]({url})\n{snippet}\n")

        return formatted_results
