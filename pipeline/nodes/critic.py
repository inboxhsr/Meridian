"""
pipeline/nodes/critic.py — Sprint 5 / Sprint 8 (observability)

Grades the quality of retrieved context using DeepSeek.
Returns structured JSON verdict determining whether to generate, retry, or abstain.

Thresholds (from project charter):
  groundedness_score >= 0.7  AND  relevance_score >= 0.6  →  'sufficient'
  Either below AND retrieval_round < 3                    →  'retry'
  Either below AND retrieval_round >= 3                   →  'abstain'

Safe-default on parse failure: retry (conservative — never silently abstains
or generates on bad evidence).
"""

from __future__ import annotations
import json
import os

from openai import OpenAI

from observability.db import init_db, log_node
from pipeline.state import MeridianState

_MAX_ROUNDS = 3
_GROUNDEDNESS_THRESHOLD = 0.7
_RELEVANCE_THRESHOLD = 0.6

_SYSTEM_PROMPT = (
    "You are a retrieval quality grader for an enterprise knowledge base.\n"
    "Given a user query and retrieved document excerpts, score the quality of the evidence.\n\n"
    "Scoring definitions:\n"
    "  groundedness (0.0–1.0): Can the query be answered from these excerpts alone? "
    "1.0 = fully answerable, 0.0 = not at all.\n"
    "  relevance (0.0–1.0): Are the excerpts on-topic for the query? "
    "1.0 = highly relevant, 0.0 = unrelated.\n\n"
    "Respond ONLY with a JSON object:\n"
    '{"groundedness_score": <float>, "relevance_score": <float>, '
    '"verdict": "sufficient"|"retry"|"abstain", "reasoning": "<one sentence>"}\n\n'
    "Verdict rules:\n"
    f"  sufficient = groundedness >= {_GROUNDEDNESS_THRESHOLD} AND relevance >= {_RELEVANCE_THRESHOLD}\n"
    "  retry      = either threshold not met (caller decides if rounds remain)\n"
    "  abstain    = evidence is fundamentally insufficient or out-of-scope\n"
    "Note: the 'retry' vs 'abstain' decision based on round count is handled by the graph, "
    "not by you. Just use 'retry' whenever the evidence is insufficient."
)

_CONTEXT_CHARS = 3_000  # keep critic prompt concise


def grade_context(state: MeridianState) -> dict:
    """LangGraph node — grade retrieved context quality.

    Reads:  safe_query, chunks, retrieval_round, query_id
    Writes: groundedness_score, relevance_score, verdict, critic_reasoning
    """
    safe_query = state["safe_query"]
    chunks = state.get("chunks", [])
    retrieval_round = state.get("retrieval_round", 1)
    query_id = state.get("query_id", "")

    if not chunks:
        # No chunks — verdict depends on remaining rounds
        verdict = "retry" if retrieval_round < _MAX_ROUNDS else "abstain"
        result = {
            "groundedness_score": 0.0,
            "relevance_score": 0.0,
            "verdict": verdict,
            "critic_reasoning": "No chunks retrieved.",
        }
        _log(query_id, result, retrieval_round, tokens_used=0)
        return result

    context_snippet = _build_snippet(chunks)
    scores, tokens_used = _grade(safe_query, context_snippet)

    # Apply threshold logic (overrides LLM verdict for consistency)
    if (
        scores["groundedness_score"] >= _GROUNDEDNESS_THRESHOLD
        and scores["relevance_score"] >= _RELEVANCE_THRESHOLD
    ):
        verdict = "sufficient"
    elif retrieval_round >= _MAX_ROUNDS:
        verdict = "abstain"
    else:
        verdict = "retry"

    result = {
        "groundedness_score": scores["groundedness_score"],
        "relevance_score":    scores["relevance_score"],
        "verdict":            verdict,
        "critic_reasoning":   scores["reasoning"],
    }
    _log(query_id, result, retrieval_round, tokens_used=tokens_used)
    return result


def _log(query_id: str, result: dict, retrieval_round: int, tokens_used: int) -> None:
    """Fire-and-forget observability log — never raises."""
    try:
        init_db()
        log_node(
            query_id=query_id,
            node_name="critic",
            model_used="deepseek-v4-flash",
            estimated_cost=0.0,     # DeepSeek free tier
            retrieval_round=retrieval_round,
            groundedness_score=result["groundedness_score"],
            relevance_score=result["relevance_score"],
            verdict=result["verdict"],
            critic_reasoning=result["critic_reasoning"],
            tokens_used=tokens_used,
        )
    except Exception:
        pass


def _build_snippet(chunks: list[dict]) -> str:
    """Build a truncated context snippet for the critic prompt."""
    parts = []
    total = 0
    for i, chunk in enumerate(chunks):
        snippet = f"[{i + 1}] {chunk.get('source_file', '?')} — {chunk.get('text', '')}"
        if total + len(snippet) > _CONTEXT_CHARS:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n".join(parts)


def _grade(safe_query: str, context_snippet: str) -> tuple[dict, int]:
    """Call DeepSeek to grade context quality. Return (scores_dict, tokens_used)."""
    _safe_default = {
        "groundedness_score": 0.0,
        "relevance_score": 0.0,
        "reasoning": "parse error — conservative retry",
    }

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return _safe_default, 0

    try:
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        user_content = f"Query: {safe_query}\n\nExcerpts:\n{context_snippet}"
        resp = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        tokens_used = (
            getattr(resp.usage, "prompt_tokens", 0) or 0
        ) + (getattr(resp.usage, "completion_tokens", 0) or 0)
        return {
            "groundedness_score": float(data.get("groundedness_score", 0.0)),
            "relevance_score":    float(data.get("relevance_score", 0.0)),
            "reasoning":          str(data.get("reasoning", "")),
        }, tokens_used
    except Exception:
        return _safe_default, 0
