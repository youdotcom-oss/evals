from abc import ABC, abstractmethod
import asyncio
import logging
import sys
import time
import traceback
from typing import Any, Dict

import httpx

from simpleqa.processing.synthesize_answer import SynthesizeAnswer


class BaseSampler(ABC):
    """Base class for all samplers"""

    def __init__(
        self,
        sampler_name: str,
        api_key: str,
        max_retries: int = 3,
        timeout: float = 60.0,
        num_results: int = 5,
        custom_args: Dict[str, Any] | None = None,
    ):
        self.sampler_name = sampler_name
        if api_key:
            self.api_key = api_key
        else:
            raise ValueError(f'API Key for sampler "{sampler_name}" is not set')
        self.max_retries = max_retries
        self.timeout = timeout
        self.num_results = num_results
        self.custom_args = custom_args

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self._setup_logger()

        self.client = self._get_client()

    def _setup_logger(self):
        """Set up logger, set logging level, and disable noisy loggers"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Disable noisy third-party logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    def _get_client(self) -> httpx.AsyncClient:
        """Setup async HTTP client with provider-specific headers"""
        base_url = self._get_base_url()
        headers = self._get_headers()
        return httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=self.timeout,
        )

    @staticmethod
    @abstractmethod
    def _get_base_url():
        """Get provider specific base url"""
        pass

    @abstractmethod
    def _get_headers(self) -> Dict[str, str]:
        """Get provider specific headers"""
        pass

    @abstractmethod
    def _get_payload(self, query: str, **kwargs) -> Dict[str, Any]:
        """Get provider specific request payload"""
        pass

    @staticmethod
    @abstractmethod
    def _get_endpoint() -> Dict[str, str]:
        """Get provider specific API endpoint"""
        pass

    @staticmethod
    @abstractmethod
    def _get_method() -> Dict[str, str]:
        """Get provider specific HTTP method"""
        pass

    @staticmethod
    @abstractmethod
    def __format_context__(results: Any) -> str:
        """Format search results into context string"""
        pass

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

    @property
    @abstractmethod
    def needs_synthesis(self) -> bool:
        """Whether this provider needs response synthesis"""
        pass

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

    async def _get_search_results(self, method, endpoint, payload):
        """Get raw search results and response time from the API"""
        start_time = time.time()
        if method == "POST":
            response = await self.client.post(endpoint, json=payload)
        elif method == "GET":
            response = await self.client.get(endpoint, params=payload)
        else:
            raise ValueError(
                'Unsupported method, please select between ["POST", "GET"]'
            )

        end_time = time.time()
        data = response.json()
        response_time_ms = round((end_time - start_time) * 1000)

        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Error {response.status_code}: {data.get('error', 'Unknown error')}",
                request=response.request,
                response=response,
            )

        return data, response_time_ms

    @staticmethod
    def __extract_query_from_messages__(message_list: list[Dict[str, Any]]) -> str:
        """Extract query from message list"""
        if isinstance(message_list, list) and len(message_list) > 0:
            last_message = message_list[-1]
            if isinstance(last_message, dict) and "content" in last_message:
                return last_message["content"]
        return str(message_list)

    @staticmethod
    async def __evaluate_response(
        query: str, ground_truth: str, generated_answer: str
    ) -> Dict[str, Any]:
        """Evaluate the generated response against ground truth"""
        from simpleqa.processing.evaluate_answer import AnswerGrader

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

        if self.custom_args:
            payload = self._get_payload(query=query, custom_args=self.custom_args)
        else:
            payload = self._get_payload(query=query)

        method = self._get_method()
        endpoint = self._get_endpoint()

        # Get raw results
        try:
            raw_results, response_time_no_retries = (
                await self._retry_with_backoff_async(
                    self._get_search_results,
                    method=method,
                    endpoint=endpoint,
                    payload=payload,
                )
            )
            formatted_results = self.__format_context__(raw_results)
        except Exception as e:
            raw_results, response_time_no_retries, formatted_results = "FAILED", "FAILED", "FAILED"
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
                raise ValueError(f"Ground truth is missing")
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

    async def close(self):
        """Cleanup resources"""
        if hasattr(self, "client"):
            await self.client.aclose()
