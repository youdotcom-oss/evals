"""Run evals using the Exa SDK"""
from typing import Any, Dict

from exa_py import Exa

from evals.samplers.base_samplers.base_sdk_sampler import BaseSDKSampler


class ExaSampler(BaseSDKSampler):
    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        num_results: int = 5,
        max_concurrency: int = 10,
        needs_synthesis: bool = True,
        custom_args: Dict[str, Any] | None = None,
    ):
        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            num_results=num_results,
            max_concurrency=max_concurrency,
            needs_synthesis=needs_synthesis,
            custom_args=custom_args,
        )

    def _initialize_client(self):
        self.client = Exa(self.api_key)

    def _get_search_results_impl(self, query: str) -> Any:
        if self.custom_args and self.custom_args["text"]:
            return self.client.search(
                query=query,
                num_results=5,
                contents={
                    "text": True
                }
            )

        raise ValueError("Unknown configuration for Exa")

    def format_results(self, results: Any) -> str:
        formatted_results = []
        raw_results = getattr(results, "results", None)

        for result in raw_results:
            if isinstance(result, dict):
                title = getattr(result, "title", "")
                url = getattr(result, "url", "")
                text = getattr(result, "text", "")
                if text:
                    formatted_results.append(f"[{title}]({url})\ntext: \"{text}\"\n")

        return "\n---\n".join(formatted_results)
