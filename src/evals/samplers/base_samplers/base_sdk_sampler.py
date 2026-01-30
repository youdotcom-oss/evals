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
        max_concurrency: int = 10,
        needs_synthesis: bool = True,
        custom_args=None,
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
        self.client = None
        if self.api_key:
            self._initialize_client()
        else:
            raise ValueError(f"API key not provided for sampler {sampler_name}. Ensure .env file is configured and contains necessary API keys")

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
