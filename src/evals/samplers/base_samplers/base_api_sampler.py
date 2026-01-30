from abc import abstractmethod
from typing import Any, Dict

import requests

from evals.samplers.base_samplers.base_sampler import BaseSampler


class BaseAPISampler(BaseSampler):
    """Base class for API-based samplers that make HTTP requests"""

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

    def _set_params(self):
        """Set API parameters before making a request"""
        self.base_url = self._get_base_url()
        self.method = self._get_method()
        self.headers = self._get_headers()
        self.endpoint = self._get_endpoint()

    @staticmethod
    @abstractmethod
    def _get_base_url() -> str:
        """Get provider specific base url"""
        pass

    @abstractmethod
    def _get_headers(self) -> Dict[str, str]:
        """Get provider specific headers"""
        pass

    @abstractmethod
    def _get_payload(self, query: str) -> Dict[str, Any]:
        """Get provider specific request payload"""
        pass

    @staticmethod
    @abstractmethod
    def _get_endpoint() -> str:
        """Get provider specific API endpoint"""
        pass

    @staticmethod
    @abstractmethod
    def _get_method() -> str:
        """Get provider specific HTTP method"""
        pass

    def get_search_results(self, query: str) -> Any:
        """Get raw search results from the API"""
        try:
            self._set_params()
            payload = self._get_payload(query)

            if self.method == "POST":
                response = requests.post(
                    self.base_url + self.endpoint,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout,
                )
            elif self.method == "GET":
                response = requests.get(
                    self.base_url + self.endpoint,
                    params=payload,
                    headers=self.headers,
                    timeout=self.timeout,
                )
            else:
                raise ValueError(
                    'Unsupported method, please select between ["POST", "GET"]'
                )

            response.raise_for_status()
            data = response.json()

            return data
        except Exception as e:
            print(f"{self.sampler_name} failed with error {e}")
            raise e

    # async def __call__(
    #         self, query_input, ground_truth: str = "", overwrite: bool = False
    # ) -> Dict[str, Any]:
    #     """Main execution pipeline"""
    #
    #     if isinstance(query_input, list):
    #         query = self.__extract_query_from_messages__(query_input)
    #     else:
    #         query = str(query_input)
    #
    #     # Get raw results
    #     try:
    #         # Run synchronous SDK call in thread pool
    #         start_time = time.time()
    #         raw_results = await asyncio.to_thread(
    #             self.get_search_results,
    #             query
    #         )
    #         response_time_no_retries = (time.time() - start_time) * 1000  # Convert to ms
    #         formatted_results = self.format_results(raw_results)
    #     except Exception as e:
    #         raw_results, response_time_no_retries, formatted_results = (
    #             "FAILED",
    #             "FAILED",
    #             "FAILED",
    #         )
    #         # TODO: Remove
    #         breakpoint()
    #         logging.exception(e)
    #
    #     # Synthesize raw results
    #     try:
    #         if self.needs_synthesis:
    #             generated_answer = await self.__synthesize_response(
    #                 query, formatted_results
    #             )
    #         else:
    #             generated_answer = formatted_results  # Already synthesized by API
    #     except Exception as e:
    #         generated_answer = "FAILED"
    #         logging.exception(e)
    #
    #     # Evaluated synthesized results against ground truth
    #     try:
    #         if ground_truth:
    #             evaluation_result_dict = await self.__evaluate_response(
    #                 query, ground_truth, generated_answer
    #             )
    #             evaluation_result = evaluation_result_dict["score_name"]
    #         else:
    #             raise ValueError("Ground truth is missing")
    #     except Exception as e:
    #         evaluation_result = "FAILED"
    #         logging.exception(e)
    #
    #     # Format result
    #     result = {
    #         "query": query,
    #         "response_time_ms": response_time_no_retries,
    #         "evaluation_result": evaluation_result,
    #         "generated_answer": generated_answer,
    #         "ground_truth": ground_truth,
    #         "raw_results": raw_results,
    #         "formatted_results": formatted_results,
    #     }
    #     return result