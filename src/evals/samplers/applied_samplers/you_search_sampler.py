"""Run evals using the you.com Search SDK https://docs.you.com/api-reference/search/v1-search"""

from typing import Any, Dict

from youdotcom import You

from evals.samplers.base_samplers.base_sdk_sampler import BaseSDKSampler


class YouSearchSnippetsSampler(BaseSDKSampler):
    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_concurrency: int = 10,
        needs_synthesis: bool = True,
    ):
        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            max_concurrency=max_concurrency,
            needs_synthesis=needs_synthesis,
        )

    def _initialize_client(self):
        self.client = You(self.api_key)

    def _get_search_results_impl(self, query: str) -> Any:
        return self.client.search.unified(
            query=query,
            count=10,
        )

    def format_results(self, results: Any) -> list[str]:
        formatted_results = []
        raw_results = []
        if results.results and results.results.web:
            raw_results.extend(results.results.web)
        if results.results and results.results.news:
            raw_results.extend(results.results.news)

        for result in raw_results:
            title = getattr(result, "title", "")
            url = getattr(result, "url", "")
            description = getattr(result, "description", "")
            snippets = getattr(result, "snippets", "")
            if snippets and isinstance(snippets, list):
                snippets = " ".join(snippets)
            formatted_results.append(f"[{title}]({url})\n snippets: {snippets}\n description: {description}")
        return formatted_results
