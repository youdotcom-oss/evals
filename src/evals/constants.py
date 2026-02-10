# We used a weaker model for synthesis and a stronger model for grading to ensure fairness.
SYNTHESIS_MODEL = "gpt-5-nano"
GRADER_MODEL = "gpt-5-mini"

# Maximum tokens available for search results (leaving room for prompt and response)
MAX_SEARCH_RESULT_TOKENS = 127750

SYNTHESIS_PROMPT = """
    You are an AI assistant that answers questions using search results.
    Read the provided search snippets carefully and answer based only on information found in the snippets.
    Keep your response clear and concise.
"""
