"""
This class is used to synthesize the results from the sampler into a concise answer. This is needed to synthesize long
search results into a single answer to be compared against the ground truth. Using the same prompt and model for all
samplers ensures an equal playing field and an apples to apples comparison across all samplers.

To view or edit the model used for synthesis, see evals.simpleqa.constants
"""

import asyncio
from dataclasses import dataclass
import logging
import os
from typing import List, Dict, Any

import httpx

from evals import constants


@dataclass
class SynthesizeAnswerResponse:
    response_text: str
    actual_queried_message_list: List[str]
    response_metadata: Dict[str, Any]


class SynthesizeAnswer:
    def __init__(self, max_retries: int = 3):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.max_retries = max_retries
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def process_single(self, session: httpx.AsyncClient, query: str, snippets: str) -> SynthesizeAnswerResponse:
        """Synthesize a single response asynchronously"""
        for trial in range(self.max_retries + 1):
            try:
                synthesis_prompt = """
                    You are an AI assistant that answers questions using search results.
                    Read the provided search snippets carefully and answer based only on information found in the snippets.
                    Keep your response clear and concise.
                """

                payload = {
                    "model": constants.SYNTHESIS_MODEL,
                    "messages": [
                        {"role": "system", "content": synthesis_prompt},
                        {
                            "role": "user",
                            "content": f"Query: {query}\n\nSearch results: {snippets}",
                        },
                    ],
                }

                response = await session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                )
                if response.status_code == 200:
                    result = response.json()
                    return SynthesizeAnswerResponse(
                        response_text=result["choices"][0]["message"]["content"],
                        actual_queried_message_list=[snippets],
                        response_metadata={
                            "model": constants.SYNTHESIS_MODEL,
                            "trial": trial,
                        },
                    )
                else:
                    error_text = response.text
                    raise Exception(f"API error {response.status_code}: {error_text}")

            except Exception as e:
                if trial >= self.max_retries:
                    self.logger.error(f"Failed after {self.max_retries} retries: {e}")
                    raise

                backoff = 2**trial
                self.logger.warning(f"Retry {trial + 1} in {backoff}s: {e}")
                await asyncio.sleep(backoff)

        raise ValueError("Could not synthesize answer")
