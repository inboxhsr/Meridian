"""
pipeline/nodes/generator.py — Sprint 5

Thin wrapper around pipeline.generator.generate() for the LangGraph node interface.
No new prompt logic — reuses the Sprint 3 generator directly.
"""

from __future__ import annotations

from pipeline.state import MeridianState
import pipeline.generator as _base_generator


def generate_answer(state: MeridianState) -> dict:
    """LangGraph node — generate a grounded answer from retrieved chunks.

    Reads:  safe_query, chunks, lang
    Writes: answer, sources, chunks_used, abstained
    """
    safe_query = state["safe_query"]
    chunks = state.get("chunks", [])
    lang = state.get("lang", "en")

    result = _base_generator.generate(safe_query, chunks, lang=lang)

    return {
        "answer":      result["answer"],
        "sources":     result["sources"],
        "chunks_used": result["chunks_used"],
        "abstained":   False,
    }
