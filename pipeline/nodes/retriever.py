"""
pipeline/nodes/retriever.py — Sprint 5

Upgraded retriever node for the LangGraph pipeline.

Handles multi-question retrieval:
  - Retrieves top_k chunks per sub-question
  - Pools all chunks across sub-questions
  - Deduplicates by (source_file, chunk_index)
  - Increments retrieval_round

Reuses pipeline.retriever.retrieve() from Sprint 3 under the hood.
"""

from __future__ import annotations

from pipeline.state import MeridianState
import pipeline.retriever as _base_retriever


def retrieve(state: MeridianState) -> dict:
    """LangGraph node — retrieve chunks for all sub-questions.

    Reads:  sub_questions, safe_query, bu_filter, top_k, retrieval_round
    Writes: chunks, retrieval_round
    """
    safe_query = state["safe_query"]
    bu_filter = state.get("bu_filter", "")
    top_k = state.get("top_k", 5)
    sub_questions = state.get("sub_questions", [])
    current_round = state.get("retrieval_round", 0)

    # If no sub-questions set, use the safe query directly
    queries = sub_questions if sub_questions else [safe_query]

    all_chunks: list[dict] = []
    seen: set[tuple] = set()

    for q in queries:
        hits = _base_retriever.retrieve(q, top_k=top_k, bu_filter=bu_filter)
        for chunk in hits:
            key = (chunk.get("source_file", ""), chunk.get("chunk_index", chunk.get("text", "")[:50]))
            if key not in seen:
                seen.add(key)
                all_chunks.append(chunk)

    return {
        "chunks": all_chunks,
        "retrieval_round": current_round + 1,
    }
