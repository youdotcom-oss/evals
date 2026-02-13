from abc import ABC, abstractmethod
import asyncio
import logging
import time
from typing import Any, Dict

from evals.configs import datasets
from evals.processing import synthesizer_utils


class BaseSampler(ABC):
    """Base class for all samplers with common functionality"""

    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_concurrency: int = 10,
        needs_synthesis: bool = True,
    ):
        self.sampler_name = sampler_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrency = max_concurrency
        self.needs_synthesis = needs_synthesis

        if api_key:
            self.api_key = api_key
        else:
            # You do not want to raise an error here, or else you can not run an eval without ALL env variables
            print(f'API Key for sampler "{sampler_name}" is not set')
            self.api_key = None

    @abstractmethod
    def get_search_results(self, query: str) -> Any:
        """
        Get raw search results from the API or SDK.

        Args:
            query: The search query string

        Returns:
            Raw search results in provider-specific format
        """
        pass

    @abstractmethod
    def format_results(self, results: Any) -> list[str]:
        """
        Format search results.

        Args:
            results: Raw search results from get_search_results

        Returns:
            tuple: (formatted_results) where formatted_results is either:
                - str: Already synthesized answer (no further synthesis needed)
                - list[str]: List of individual search results (needs synthesis)
        """
        pass

    @staticmethod
    def __extract_query_from_messages__(message_list: list[dict]) -> str:
        """Extract query from message list"""
        if isinstance(message_list, list) and len(message_list) > 0:
            last_message = message_list[-1]
            if isinstance(last_message, dict) and "content" in last_message:
                return last_message["content"]
        return str(message_list)

    @staticmethod
    async def __evaluate_response(query: str, ground_truth: str, generated_answer: str, dataset: datasets.Dataset) -> Dict[str, Any]:
        """Evaluate the generated response against ground truth"""
        return await dataset.grader(query, ground_truth, generated_answer)

    async def __call__(self, query_input, dataset: dict, ground_truth: str = "", overwrite: bool = False) -> Dict[str, Any]:
        """Main execution pipeline"""
        internal_response_time_ms = None
        end_to_end_time_ms = None
        if isinstance(query_input, list):
            query = self.__extract_query_from_messages__(query_input)
        else:
            query = str(query_input)

        end_to_end_start_time = time.time()
        # Get raw results
        try:
            # Run synchronous SDK call in thread pool
            raw_results = await asyncio.to_thread(self.get_search_results, query)
            if self.sampler_name == 'you_search_livecrawl':
                internal_response_time_ms = round(raw_results["metadata"]["latency"] * 1000, 2)  # Convert to ms
            elif self.sampler_name == 'you_search':
                internal_response_time_ms = round(raw_results.metadata.latency * 1000, 2)  # Convert to ms
            elif 'tavily' in self.sampler_name:
                internal_response_time_ms = round(raw_results['response_time'] * 1000, 2)  # Convert to ms
            elif 'exa' in self.sampler_name:
                # Exa does not return internal run time, best we can do is API call time
                internal_response_time_ms = round((time.time() - end_to_end_start_time) * 1000, 2)  # Convert to ms

            formatted_results = self.format_results(raw_results)
        except Exception as e:
            raw_results, internal_response_time_ms, end_to_end_time_ms, formatted_results = (
                "FAILED",
                "FAILED",
                "FAILED",
                "FAILED",
            )
            logging.exception(e)

        # Synthesize raw results
        try:
            if self.needs_synthesis:
                generated_answer = synthesizer_utils.synthesize_response(query, formatted_results)
            else:
                generated_answer = formatted_results  # Already synthesized by API

            end_to_end_end_time = time.time()
            end_to_end_time_ms = round((end_to_end_end_time - end_to_end_start_time) * 1000, 2)
        except Exception as e:
            generated_answer = "FAILED"
            logging.exception(e)

        # Evaluated synthesized results against ground truth
        try:
            if ground_truth:
                evaluation_result_dict = await self.__evaluate_response(query, ground_truth, generated_answer, dataset)
                evaluation_result = evaluation_result_dict["score_name"]
            else:
                raise ValueError("Ground truth is missing")
        except Exception as e:
            evaluation_result = "FAILED"
            logging.exception(e)

        # Format result
        result = {
            "query": query,
            "internal_response_time_ms": internal_response_time_ms,
            "end_to_end_time_ms": end_to_end_time_ms,
            "evaluation_result": evaluation_result,
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "raw_results": raw_results,
            "formatted_results": formatted_results,
        }
        return result
