import os
from typing import Any, Dict

from evals.samplers.base_samplers.base_api_sampler import BaseAPISampler


class SerpApiGoogleSampler(BaseAPISampler):
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
        return "https://serpapi.com"

    @staticmethod
    def _get_endpoint() -> str:
        return "/search"

    @staticmethod
    def _get_method() -> str:
        return "GET"

    def _get_headers(self) -> Dict[str, str]:
        return {}

    def _get_payload(self, query: str, custom_args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return {
            "q": query,
            "engine": "google",
            "num": 5,
            "api_key": self.api_key,
        }

    def format_results(self, results: Any) -> str:
        formatted_results = []
        if "organic_results" in results:
            for result in results["organic_results"]:
                if isinstance(result, dict):
                    title = result.get("title", "")
                    link = result.get("link", "")
                    snippet = result.get("snippet", "")
                    if snippet and isinstance(snippet, list):
                        snippet = " ".join(snippet)
                    formatted_results.append(f"[{title}]({link})\n snippet: {snippet}")
        return "\n---\n".join(formatted_results)
