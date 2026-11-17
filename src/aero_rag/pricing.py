"""Model prices used for cost estimates (evaluation) and per-call cost in traces (Langfuse).

USD per 1M tokens. Langfuse is given the cost explicitly because its own
price table may not know the Azure deployment name.
"""

LLM_INPUT_PRICE = 0.75  # gpt-5.4-mini, input
LLM_OUTPUT_PRICE = 4.50  # gpt-5.4-mini, output
EMBED_PRICE = 0.02  # text-embedding-3-small


def llm_cost(prompt_tokens: int, completion_tokens: int) -> dict[str, float]:
    """Cost of one chat call, split the way Langfuse expects (input / output / total)."""
    cost_in = prompt_tokens * LLM_INPUT_PRICE / 1e6
    cost_out = completion_tokens * LLM_OUTPUT_PRICE / 1e6
    return {"input": cost_in, "output": cost_out, "total": cost_in + cost_out}
