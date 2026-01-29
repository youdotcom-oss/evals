from abc import abstractmethod
import asyncio
import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from evals.samplers.base_samplers.base_sampler import BaseSampler


class BaseSDKSampler(BaseSampler):
    """Base class for SDK-based samplers that use provider SDKs"""

    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        num_results: int = 5,
        max_concurrency: int = 10,
        needs_synthesis: bool = True,
        custom_args=None,
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
        self.client = None
        if self.api_key:
            self._initialize_client()
        else:
            raise ValueError("API key not provided")

    @abstractmethod
    def _initialize_client(self):
        """
        Initialize the SDK client with the API key.

        Returns:
            Initialized SDK client instance
        """
        pass

    @abstractmethod
    def _get_search_results_impl(self, query: str) -> Any:
        """
        Implementation of getting raw search results using the SDK client.
        This method should be implemented by derived classes.

        Args:
            query: The search query string

        Returns:
            Raw search results in provider-specific format
        """
        pass

    def get_search_results(self, query: str) -> Any:
        """
        Get raw search results using the SDK client.
        This method wraps _get_search_results_impl with error handling and timeout.

        Args:
            query: The search query string

        Returns:
            Raw search results in provider-specific format

        Raises:
            TimeoutError: If the search operation exceeds the timeout
            Exception: Re-raises any exception encountered during search
        """
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._get_search_results_impl, query)
                return future.result(timeout=self.timeout)
        except TimeoutError:
            error_msg = f"{self.sampler_name} timed out after {self.timeout} seconds"
            logging.error(error_msg)
            raise TimeoutError(error_msg)
        except Exception as e:
            logging.error(f"{self.sampler_name} failed with error {e}")
            raise e

    async def _retry_with_backoff_async(self, func, *args, **kwargs):
        """Generic async retry logic with exponential backoff"""
        trial = 0
        while True:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                _, _, traceback_ = sys.exc_info()
                if trial >= self.max_retries:
                    logging.error(f"Failed after {self.max_retries} retries: {str(e)}")
                    raise

                trial += 1
                backoff_time = 2**trial
                logging.warning(
                    f"Attempt {trial}/{self.max_retries} failed: {traceback.print_tb(traceback_)}. Retrying in {backoff_time}s..."
                )
                await asyncio.sleep(backoff_time)

    async def __call__(
            self, query_input, ground_truth: str = "", overwrite: bool = False
    ) -> Dict[str, Any]:
        """Main execution pipeline"""

        if isinstance(query_input, list):
            query = self.__extract_query_from_messages__(query_input)
        else:
            query = str(query_input)

        # if self.custom_args:
        #     payload = self._get_payload(query=query, custom_args=self.custom_args)
        # else:
        #     payload = self._get_payload(query=query)
        #
        # method = self._get_method()
        # endpoint = self._get_endpoint()

        # Get raw results
        try:
            # Run synchronous SDK call in thread pool
            start_time = time.time()
            raw_results = await asyncio.to_thread(
                self.get_search_results,
                query
            )
            response_time_no_retries = (time.time() - start_time) * 1000  # Convert to ms
            formatted_results = self.format_results(raw_results)
        except Exception as e:
            raw_results, response_time_no_retries, formatted_results = (
                "FAILED",
                "FAILED",
                "FAILED",
            )
            breakpoint()
            logging.exception(e)

        # Synthesize raw results
        try:
            if self.needs_synthesis:
                generated_answer = await self.__synthesize_response(
                    query, formatted_results
                )
            else:
                generated_answer = formatted_results  # Already synthesized by API
        except Exception as e:
            generated_answer = "FAILED"
            logging.exception(e)

        # Evaluated synthesized results against ground truth
        try:
            if ground_truth:
                evaluation_result_dict = await self.__evaluate_response(
                    query, ground_truth, generated_answer
                )
                evaluation_result = evaluation_result_dict["score_name"]
            else:
                raise ValueError("Ground truth is missing")
        except Exception as e:
            evaluation_result = "FAILED"
            logging.exception(e)

        # Format result
        result = {
            "query": query,
            "response_time_ms": response_time_no_retries,
            "evaluation_result": evaluation_result,
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "raw_results": raw_results,
            "formatted_results": formatted_results,
        }
        return result

    # async def close(self):
    #     """Cleanup resources"""
    #     if hasattr(self, "client"):
    #         await self.client.close()
