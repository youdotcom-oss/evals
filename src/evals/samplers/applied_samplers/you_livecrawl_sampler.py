import random
from typing import Any, Dict, List

from evals.samplers.base_samplers.base_api_sampler import (
    BaseAPISampler,
)


class YouLivecrawlSampler(BaseAPISampler):
    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_concurrency: int = 10,
        needs_synthesis: bool = True,
        custom_args: Dict[str, Any] | None = None,
    ):
        if api_key is None:
            raise ValueError(
                f"API key not provided for sampler {sampler_name}. Ensure .env file is configured and contains necessary API keys"
            )

        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            max_concurrency=max_concurrency,
            needs_synthesis=needs_synthesis,
            custom_args=custom_args,
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
            # These parameters are in beta, and are designed to maximize performance
            "num_bytes": 500000 + random.randint(1, 100),
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
