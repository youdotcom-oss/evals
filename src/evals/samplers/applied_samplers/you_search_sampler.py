from typing import Any, Dict

import youdotcom

from evals.samplers.base_samplers.base_api_sampler import (
    BaseAPISampler,
)
from evals.samplers.base_samplers.base_sdk_sampler import (
    BaseSDKSampler,
)


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
        self.client = youdotcom.You(self.api_key)

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
            formatted_results.append(
                f"[{title}]({url})\n snippets: {snippets}\n description: {description}"
            )
        return formatted_results


class YouLivecrawlSampler(BaseAPISampler):
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

    @staticmethod
    def _get_base_url():
        return "https://ydc-index.io"

    def _get_headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key}

    @staticmethod
    def _get_method() -> str:
        return "GET"

    @staticmethod
    def _get_endpoint() -> str:
        return "/v1/search/"

    def _get_payload(self, query: str) -> Dict[str, Any]:
        return {
            "query": query,
            "count": 10,
            "livecrawl": "all",
            "livecrawl_formats": "markdown",
            # These parameters are in beta and are designed to maximize performance
            "num_bytes": 500000,
            "crawl_timeout": 1,
        }

    def format_results(self, results: Any) -> list[str]:
        formatted_results = []
        if "results" in results:
            if "web" not in results["results"]:
                return [""]

            if "news" in results["results"]:
                all_results = results["results"]["news"] + results["results"]["web"]
            else:
                all_results = results["results"]["web"]

            for result in all_results:
                title = result.get("title", "")
                url = result.get("url", "")
                contents = result.get("contents", "")

                if "markdown" in contents:
                    contents = contents["markdown"]
                    formatted_result = f"[{title}]({url})\n{contents}"
                    formatted_results.append(formatted_result)
                else:
                    description = result.get("description", "")
                    snippet = result.get("snippets", "")
                    if snippet and isinstance(snippet, list):
                        snippet = " ".join(snippet)
                    formatted_result = f"[{title}]({url})\n snippet: {snippet}\n description: {description}"
                    formatted_results.append(formatted_result)

        return formatted_results
