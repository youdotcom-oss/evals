"""Run evals using the Tavily SDK"""

from typing import Any, Dict

from tavily import TavilyClient

from evals.samplers.base_samplers.base_sdk_sampler import BaseSDKSampler


class TavilySampler(BaseSDKSampler):
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

    def _initialize_client(self):
        self.client = TavilyClient(self.api_key)

    def _get_search_results_impl(self, query: str) -> Any:
        if self.custom_args and self.custom_args["search_depth"]:
            return self.client.search(
                query=query,
                max_results=5,
                search_depth=self.custom_args["search_depth"],
            )
        raise ValueError("Unknown configuration for Tavily")

    def format_results(self, results: Any) -> str:
        formatted_results = []
        raw_results = results["results"]

        for result in raw_results:
            if isinstance(result, dict):
                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", "")
                if content:
                    formatted_results.append(f"[{title}]({url})\ncontent: {content}\n")

        return "\n---\n".join(formatted_results)
