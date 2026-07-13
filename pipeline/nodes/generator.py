"""
pipeline/nodes/generator.py — Sprint 5 / Sprint 8 (observability)

Thin wrapper around pipeline.generator.generate() for the LangGraph node interface.
No new prompt logic — reuses the Sprint 3 generator directly.
"""

from __future__ import annotations

from observability.db import init_db, log_node
from pipeline.state import MeridianState
import pipeline.generator as _base_generator


def generate_answer(state: MeridianState) -> dict:
    """LangGraph node — generate a grounded answer from retrieved chunks.

    Reads:  safe_query, chunks, lang, query_id
    Writes: answer, sources, chunks_used, abstained
    """
    safe_query = state["safe_query"]
    chunks = state.get("chunks", [])
    lang = state.get("lang", "en")
    query_id = state.get("query_id", "")

    result = _base_generator.generate(safe_query, chunks, lang=lang)

    # ── Observability log ─────────────────────────────────────────────────────
    try:
        init_db()
        log_node(
            query_id=query_id,
            node_name="generator",
            model_used="deepseek-v4-flash",
            estimated_cost=0.0,                     # DeepSeek free tier
            tokens_used=len(result.get("answer", "")),  # char count proxy
        )
    except Exception:
        pass

    return {
        "answer":      result["answer"],
        "sources":     result["sources"],
        "chunks_used": result["chunks_used"],
        "abstained":   False,
    }
