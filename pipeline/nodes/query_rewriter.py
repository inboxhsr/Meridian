"""
pipeline/nodes/query_rewriter.py — Sprint 5 / Sprint 8 (observability)

Decomposes a user query into one or more atomic sub-questions using DeepSeek.
For simple queries a single-element list is returned (the original query).

Key constraints:
  - Never translates — stays in the query's original language.
  - Always returns at least one sub-question.
  - Safe-default on parse failure: [safe_query].
"""

from __future__ import annotations
import json
import os

from openai import OpenAI

from observability.db import init_db, log_node
from pipeline.state import MeridianState

_SYSTEM_PROMPT = (
    "You are a query decomposition assistant for an enterprise knowledge base.\n"
    "Break the user's question into clear, atomic sub-questions that can each be answered "
    "with a single document search.\n\n"
    "Rules:\n"
    "  - Never translate. Keep the exact language of the original query.\n"
    "  - For a simple, atomic question return a JSON array with exactly one item.\n"
    "  - For a compound or comparative question return two or more items.\n"
    "  - Each sub-question must be a complete, standalone question.\n\n"
    'Respond ONLY with a JSON array of strings, e.g. ["What is X?", "What is Y?"]'
)


def rewrite_query(state: MeridianState) -> dict:
    """LangGraph node — decompose safe_query into sub-questions.

    Reads:  safe_query, query_id, retrieval_round
    Writes: sub_questions
    """
    safe_query = state["safe_query"]
    query_id = state.get("query_id", "")
    retrieval_round = state.get("retrieval_round", 0)

    sub_questions, tokens_used = _decompose_with_usage(safe_query)

    # ── Observability log ─────────────────────────────────────────────────────
    try:
        init_db()
        log_node(
            query_id=query_id,
            node_name="query_rewriter",
            model_used="deepseek-v4-flash",
            tokens_used=tokens_used,
            estimated_cost=0.0,     # DeepSeek free tier
            retrieval_round=retrieval_round,
        )
    except Exception:
        pass

    return {"sub_questions": sub_questions}


def _decompose(safe_query: str) -> list[str]:
    """Call DeepSeek to decompose the query. Return safe-default on failure.

    Public interface kept for backward-compatibility with existing tests.
    """
    result, _ = _decompose_with_usage(safe_query)
    return result


def _decompose_with_usage(safe_query: str) -> tuple[list[str], int]:
    """Internal: decompose and return (sub_questions, total_tokens)."""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return [safe_query], 0

    try:
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": safe_query},
            ],
            max_tokens=300,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        sub_questions = json.loads(raw)
        if (
            isinstance(sub_questions, list)
            and sub_questions
            and all(isinstance(q, str) and q.strip() for q in sub_questions)
        ):
            total_tokens = (
                getattr(resp.usage, "prompt_tokens", 0) or 0
            ) + (getattr(resp.usage, "completion_tokens", 0) or 0)
            return sub_questions, total_tokens
        return [safe_query], 0
    except Exception:
        return [safe_query], 0
