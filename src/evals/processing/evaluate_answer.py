"""
This class is used to evaluate the correctness of the response. No changes have been made to the grading prompt.

To view or edit the model used for grading, see evals.simpleqa.constants
"""

import asyncio
import logging
import os
import re
from typing import Dict, Any

import httpx

from evals import constants


class AnswerGrader:
    def __init__(self, model: str = constants.GRADER_MODEL, max_retries: int = 3):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model = model
        self.max_retries = max_retries
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def call_openai_async(self, client: httpx.AsyncClient, prompt: str) -> str:
        """Make async call to OpenAI API"""
        for trial in range(self.max_retries + 1):
            try:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 1024,
                }

                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    if content is None:
                        raise ValueError("OpenAI API returned empty response")
                    return content
                else:
                    raise Exception(f"API error {response.status_code}: {response.text}")

            except Exception as e:
                if trial >= self.max_retries:
                    self.logger.error(f"Failed after {self.max_retries} retries: {e}")
                    raise

                backoff = 2**trial
                self.logger.warning(f"Evaluation retry {trial + 1} in {backoff}s: {e}")
                await asyncio.sleep(backoff)

        raise ValueError("Failed to call OpenAI API")

    async def evaluate_single_simpleqa(self, question: str, target: str, predicted_answer: str) -> Dict[str, Any]:
        """Evaluate a single response asynchronously for SimpleQA dataset"""
        grader_prompt = constants.SIMPLEQA_ANSWER_GRADER_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            grading_response = await self.call_openai_async(client, grader_prompt)

        # Parse the grade
        match = re.search(r"(A|B|C)", grading_response)
        grade_letter = match.group(0) if match else "C"

        # Convert to readable format
        score_name = {"A": "is_correct", "B": "is_incorrect", "C": "is_not_attempted"}[grade_letter]

        is_correct = grade_letter == "A"
        is_incorrect = grade_letter == "B"
        is_not_attempted = grade_letter == "C"

        return {
            "grade": grade_letter,
            "score_name": score_name,
            "is_correct": is_correct,
            "is_incorrect": is_incorrect,
            "is_not_attempted": is_not_attempted,
            "score": is_correct,
        }

    async def evaluate_single_frames(self, question: str, target: str, predicted_answer: str) -> Dict[str, Any]:
        """Evaluate a single response asynchronously for frames dataset"""
        grader_prompt = constants.FRAMES_ANSWER_GRADER_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            grading_response = await self.call_openai_async(client, grader_prompt)

        # Parse the grade
        match = re.search(r"(TRUE|FALSE)", grading_response)
        grade_letter = match.group(0) if match else None

        # Convert to readable format
        score_name = {"TRUE": "is_correct", "FALSE": "is_incorrect"}[grade_letter]

        is_correct = grade_letter == "TRUE"
        is_incorrect = grade_letter == "FALSE"

        return {
            "grade": grade_letter,
            "score_name": score_name,
            "is_correct": is_correct,
            "is_incorrect": is_incorrect,
            "score": is_correct,
        }