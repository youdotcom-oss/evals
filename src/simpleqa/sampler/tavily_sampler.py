import os
from typing import Any, Dict

from simpleqa.sampler.base_sampler import BaseSampler


class TavilySampler(BaseSampler):
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
            os.getenv("TAVILY_API_KEY"),
            max_retries,
            timeout,
            num_results,
            custom_args,
        )

    @staticmethod
    def _get_base_url():
        return "https://api.tavily.com/"

    @staticmethod
    def _get_endpoint() -> str:
        return "search"

    @staticmethod
    def _get_method() -> str:
        return "POST"

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _get_payload(
        self, query: str, custom_args: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        return {
            "query": query,
            "max_results": self.num_results,
        }

    @staticmethod
    def __format_context__(results: Any) -> str:
        formatted_results = []
        if "results" in results:
            for result in results["results"]:
                if isinstance(result, dict):
                    title = result.get("title", "")
                    url = result.get("url", "")
                    content = result.get("content", "")
                    formatted_results.append(f"[{title}]({url})\n{content}\n")
        return "\n---\n".join(formatted_results)
