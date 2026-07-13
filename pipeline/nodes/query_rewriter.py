"""
pipeline/nodes/query_rewriter.py — Sprint 5

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

    Reads:  safe_query
    Writes: sub_questions
    """
    safe_query = state["safe_query"]
    sub_questions = _decompose(safe_query)
    return {"sub_questions": sub_questions}


def _decompose(safe_query: str) -> list[str]:
    """Call DeepSeek to decompose the query. Return safe-default on failure."""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return [safe_query]

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
            return sub_questions
        return [safe_query]
    except Exception:
        return [safe_query]
