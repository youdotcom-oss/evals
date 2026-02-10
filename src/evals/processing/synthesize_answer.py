"""
This class is used to synthesize the results from the sampler into a concise answer. This is needed to synthesize long
search results into a single answer to be compared against the ground truth. Using the same prompt and model for all
samplers ensures an equal playing field and an apples to apples comparison across all samplers.

To view or edit the model used for synthesis, see evals.constants
"""

from dataclasses import dataclass
import logging
import os
import requests
import time
import traceback
from typing import List, Dict, Any


@dataclass
class SynthesizeAnswerResponse:
    response_text: str
    actual_queried_message_list: List[str]
    response_metadata: Dict[str, Any]


class SynthesizeAnswer:
    def __init__(self, synthesis_prompt: str, synthesis_model: str, max_retries: int = 3):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.synthesis_prompt = synthesis_prompt
        self.synthesis_model = synthesis_model
        self.max_retries = max_retries
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def process_single(self, query: str, results: str) -> SynthesizeAnswerResponse:
        """Synthesize a single response"""
        for trial in range(self.max_retries + 1):
            try:
                payload = {
                    "model": self.synthesis_model,
                    "messages": [
                        {"role": "system", "content": self.synthesis_prompt},
                        {
                            "role": "user",
                            "content": f"Query: {query}\n\nSearch results: {results}",
                        },
                    ],
                }

                import time

                start_time = time.time()
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                )

                if response.status_code == 200:
                    result = response.json()
                    return SynthesizeAnswerResponse(
                        response_text=result["choices"][0]["message"]["content"],
                        actual_queried_message_list=[results],
                        response_metadata={
                            "model": self.synthesis_model,
                            "trial": trial,
                        },
                    )
                if response.status_code == 402:
                    print("Rate limit hit")
                    # TODO: Find a clever way to cut this eval short, but not stop a long chain of evals
                    quit()
                else:
                    error_text = response.text
                    print(f"ERROR: Failed synthesis after {self.max_retries} retries")
                    traceback.print_exc()
                    raise Exception(f"API error {response.status_code}: {error_text}")

            except Exception as e:
                if trial >= self.max_retries:
                    print(f"ERROR: Failed synthesis after {self.max_retries} retries")
                    traceback.print_exc()
                    raise

                backoff = 2**trial
                print(f"WARNING: Retry {trial + 1} in {backoff}s: {e}")
                time.sleep(backoff)

        raise ValueError("Could not synthesize answer")
