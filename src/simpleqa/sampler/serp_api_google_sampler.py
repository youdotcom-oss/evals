import os
from typing import Any, Dict

from simpleqa.sampler.base_sampler import BaseSampler


class SerpApiGoogleSampler(BaseSampler):

    @property
    def needs_synthesis(self) -> bool:
        return True  # Search provider, needs answer synthesis

    def __init__(
        self,
        sampler_name: str,
        max_retries: int = 3,
        timeout: float = 60.0,
        num_results: int = 5,
        custom_args: Dict[str, Any] | None = None,
    ):
        super().__init__(
            sampler_name,
            os.getenv("SERP_API_KEY"),
            max_retries,
            timeout,
            num_results,
            custom_args,
        )

    @staticmethod
    def _get_base_url():
        return "https://serpapi.com"

    @staticmethod
    def _get_endpoint() -> str:
        return "search/"

    @staticmethod
    def _get_method() -> str:
        return "GET"

    def _get_headers(self) -> Dict[str, str]:
        return {}

    def _get_payload(
        self, query: str, custom_args: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        return {
            "q": query,
            "engine": "google",
            "num": self.num_results,
            "api_key": self.api_key,
        }

    @staticmethod
    def __format_context__(results: Any) -> str:
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
