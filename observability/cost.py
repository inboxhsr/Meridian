"""
observability/cost.py — Sprint 8

Cost calculation from token counts and model name.
DeepSeek (deepseek-v4-flash) and Gemini Embedding are free-tier — always $0.00.
Only Gemini Flash (gemini-2.5-flash) has a non-zero rate.

Rates from: https://ai.google.dev/pricing (as of 2026-07)
"""

from __future__ import annotations

# Rates per 1M tokens (USD)
_RATES: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD. Returns 0.0 for free-tier / local models."""
    rate = _RATES.get(model)
    if rate is None:
        return 0.0
    return (input_tokens * rate["input"] + output_tokens * rate["output"]) / 1_000_000
