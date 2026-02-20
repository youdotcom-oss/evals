import math
from typing import List

import tiktoken

from evals.processing.synthesize_answer import SynthesizeAnswer
from evals.constants import (
    SYNTHESIS_PROMPT,
    MAX_SEARCH_RESULT_TOKENS,
    SYNTHESIS_MODEL,
)


def trim_results_to_model_limit(
    formatted_results: list[str],
    synthesis_model: str,
) -> list[str]:
    """
    Trim search results to fit within the synthesis model's token limit.

    Args:
        formatted_results: List of strings, each representing a search result
        synthesis_model: The model name used for synthesis (e.g., "gpt-4o-mini")

    Returns:
        List of trimmed result strings that fit within token limits
    """

    # Initialize tokenizer
    enc = tiktoken.encoding_for_model(synthesis_model)

    # Sort results by length (token count) - shortest first
    results_with_tokens = [(result, enc.encode(result)) for result in formatted_results]
    results_with_tokens.sort(key=lambda x: len(x[1]))

    # Track remaining tokens available
    remaining_search_result_tokens = MAX_SEARCH_RESULT_TOKENS

    # Trim each result to fit within token limit
    trimmed_results = []
    for i, (result, tokens) in enumerate(results_with_tokens):
        # Calculate max tokens per result based on remaining tokens and remaining results
        remaining_results = len(results_with_tokens) - i
        max_tokens_per_result = math.floor(remaining_search_result_tokens / remaining_results)

        # If within limit, keep as is; otherwise truncate to max_tokens_per_result
        if len(tokens) <= max_tokens_per_result:
            trimmed_results.append(result)
            remaining_search_result_tokens -= len(tokens)
        else:
            # Truncate token list and decode back to text
            truncated_tokens = tokens[:max_tokens_per_result]
            trimmed_result = enc.decode(truncated_tokens)
            trimmed_results.append(trimmed_result)
            remaining_search_result_tokens -= len(truncated_tokens)

    return trimmed_results


async def synthesize_response(
    query: str,
    formatted_results: list[str],
    synthesis_model: str = SYNTHESIS_MODEL,
) -> str:
    """
    Async method for synthesizing responses from search results using OpenAI
    """
    # Trim results to fit within model token limits
    trimmed_results = trim_results_to_model_limit(formatted_results, synthesis_model)

    # Concatenate results with separator
    concatenated_results = "\n---\n".join(trimmed_results)

    answer_synthesizer = SynthesizeAnswer(SYNTHESIS_PROMPT, max_retries=3, synthesis_model=synthesis_model)
    result = await answer_synthesizer.process_single(query, concatenated_results)
    return result.response_text if result else f"Synthesis failed for: {query}"
