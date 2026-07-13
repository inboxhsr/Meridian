"""
pipeline/nodes/reranker.py — Sprint 6 / Sprint 8 (observability)

BGE-reranker-v2-m3 cross-encoder reranker node.

Runs ONCE on the full deduplicated chunk pool produced by the retriever.
Returns chunks reordered by cross-encoder score, trimmed to top_k.

Design:
  - Module-level model cache: loads once per process (~1.1 GB, CPU float32)
  - Scores sigmoid-normalized to [0, 1] range
  - Compound queries: retriever already deduplicates; reranker sees one merged pool
  - Cost: $0.00 — runs entirely locally, no API call

Implementation note:
  Uses sentence_transformers.CrossEncoder (not FlagReranker) — CrossEncoder uses
  the fast AutoTokenizer and is compatible with all current transformers versions.
  FlagReranker's slow XLMRobertaTokenizer lost prepare_for_model in newer
  transformers releases.
"""

from __future__ import annotations
import math

from observability.db import init_db, log_node
from pipeline.state import MeridianState

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_reranker = None  # module-level cache — loaded once on first request


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _get_reranker():
    """Load and cache the BGE reranker model (downloads on first call)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(_MODEL_NAME, max_length=512)
    return _reranker


def rerank(state: MeridianState) -> dict:
    """LangGraph node — rerank retrieved chunks by cross-encoder score.

    Reads:  chunks, safe_query, top_k, query_id
    Writes: chunks  (reordered descending by reranker_score, trimmed to top_k)
    """
    chunks = state.get("chunks", [])
    safe_query = state["safe_query"]
    top_k = state.get("top_k", 5)
    query_id = state.get("query_id", "")

    if not chunks:
        return {"chunks": []}

    reranker = _get_reranker()

    # Build (query, passage) pairs for the cross-encoder
    pairs = [[safe_query, c.get("text", "")] for c in chunks]

    # predict() returns raw logit scores; apply sigmoid to map to [0, 1]
    raw_scores = reranker.predict(pairs)
    normalized = [_sigmoid(float(s)) for s in raw_scores]

    # Attach reranker_score to each chunk dict and sort descending
    for chunk, score in zip(chunks, normalized):
        chunk["reranker_score"] = score

    reranked = sorted(chunks, key=lambda c: c["reranker_score"], reverse=True)

    # ── Observability log ─────────────────────────────────────────────────────
    try:
        init_db()
        log_node(
            query_id=query_id,
            node_name="reranker",
            estimated_cost=0.0,         # local model — no API cost
            tokens_used=len(chunks),    # number of pairs scored
        )
    except Exception:
        pass

    return {"chunks": reranked[:top_k]}
