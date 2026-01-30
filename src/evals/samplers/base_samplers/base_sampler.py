from abc import ABC, abstractmethod
import asyncio
import logging
import time
from typing import Any, Dict

import httpx

from evals.processing.synthesize_answer import SynthesizeAnswer


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
        custom_args=None,
    ):
        self.sampler_name = sampler_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrency = max_concurrency
        self.needs_synthesis = needs_synthesis
        self.custom_args = custom_args

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
    def format_results(self, results: Any) -> str:
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


    async def __synthesize_response(self, query: str, formatted_context: str) -> str:
        """
        Private method for synthesizing responses from search results using OpenAI
        """
        answer_synthesizer = SynthesizeAnswer(max_retries=3)
        async with httpx.AsyncClient(timeout=30.0) as client:
            result = await answer_synthesizer.process_single(
                client, query, formatted_context
            )
        return result.response_text if result else f"Synthesis failed for: {query}"

    @staticmethod
    async def __evaluate_response(
        query: str, ground_truth: str, generated_answer: str
    ) -> Dict[str, Any]:
        """Evaluate the generated response against ground truth"""
        from evals.processing.evaluate_answer import AnswerGrader

        evaluator = AnswerGrader()
        return await evaluator.evaluate_single(query, ground_truth, generated_answer)

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
            # TODO: Remove
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