"""
tests/test_reranker_dedup.py — Sprint 6

Tests that the reranker correctly handles unique chunks and doesn't
introduce duplicates. Deduplication happens in the retriever node;
these tests verify the reranker preserves chunk identity correctly.

All tests mock FlagReranker — no model download needed.
"""

from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock


def _make_chunk(source: str, text: str, score: float = 0.8) -> dict:
    return {
        "source_file": source,
        "text": text,
        "bu": "hr",
        "lang": "en",
        "modality": "text",
        "score": score,
        "chunk_index": 0,
    }


def _mock_reranker(scores: list[float]):
    mock = MagicMock()
    mock.predict.return_value = scores
    return mock


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_unique_chunks_in_unique_chunks_out(monkeypatch):
    """N unique chunks in must produce N (or fewer) unique chunks out — no duplication."""
    chunks = [
        _make_chunk("doc1.pdf", "Expense limit is $500"),
        _make_chunk("doc2.pdf", "Leave policy is 20 days"),
        _make_chunk("doc3.pdf", "IT access control policy"),
    ]
    with patch("pipeline.nodes.reranker._get_reranker") as mock_get:
        mock_get.return_value = _mock_reranker([0.9, 0.6, 0.3])
        from pipeline.nodes.reranker import rerank
        state = {"safe_query": "expense policy", "chunks": chunks, "top_k": 10}
        result = rerank(state)  # type: ignore[arg-type]

    source_files = [c["source_file"] for c in result["chunks"]]
    assert len(source_files) == len(set(source_files)), "Duplicate source files found in output"


def test_chunk_source_file_preserved(monkeypatch):
    """Reranker must not mutate the source_file field of chunks."""
    chunks = [
        _make_chunk("hr_expense_policy.en.pdf", "text A"),
        _make_chunk("it_security_access.en.pdf", "text B"),
    ]
    original_sources = {c["source_file"] for c in chunks}
    with patch("pipeline.nodes.reranker._get_reranker") as mock_get:
        mock_get.return_value = _mock_reranker([0.7, 0.5])
        from pipeline.nodes.reranker import rerank
        state = {"safe_query": "some query", "chunks": chunks, "top_k": 5}
        result = rerank(state)  # type: ignore[arg-type]

    result_sources = {c["source_file"] for c in result["chunks"]}
    assert result_sources.issubset(original_sources), "source_file was altered by reranker"


def test_reranker_scores_in_zero_one_range(monkeypatch):
    """With normalize=True, all reranker_scores must be in [0, 1]."""
    chunks = [_make_chunk(f"doc{i}.pdf", f"text {i}") for i in range(5)]
    # Simulate sigmoid-normalized scores
    normalized_scores = [0.95, 0.82, 0.61, 0.44, 0.23]
    with patch("pipeline.nodes.reranker._get_reranker") as mock_get:
        mock_get.return_value = _mock_reranker(normalized_scores)
        from pipeline.nodes.reranker import rerank
        state = {"safe_query": "some query", "chunks": chunks, "top_k": 10}
        result = rerank(state)  # type: ignore[arg-type]

    for chunk in result["chunks"]:
        assert 0.0 <= chunk["reranker_score"] <= 1.0, (
            f"reranker_score {chunk['reranker_score']} out of [0, 1] range"
        )
