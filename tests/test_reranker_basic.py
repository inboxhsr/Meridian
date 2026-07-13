"""
tests/test_reranker_basic.py — Sprint 6

Unit tests for pipeline/nodes/reranker.py.
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
    """Return a mock CrossEncoder whose predict returns the given scores."""
    mock = MagicMock()
    mock.predict.return_value = scores
    return mock


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_rerank_returns_chunks_key(monkeypatch):
    """Node must return a dict with 'chunks' key."""
    with patch("pipeline.nodes.reranker._get_reranker") as mock_get:
        mock_get.return_value = _mock_reranker([0.9, 0.5, 0.3])
        from pipeline.nodes.reranker import rerank
        state = {
            "safe_query": "What is the expense limit?",
            "chunks": [
                _make_chunk("a.pdf", "Expense limit is $500"),
                _make_chunk("b.pdf", "Leave policy details"),
                _make_chunk("c.pdf", "IT security incident response"),
            ],
            "top_k": 5,
        }
        result = rerank(state)  # type: ignore[arg-type]
    assert "chunks" in result


def test_rerank_trims_to_top_k(monkeypatch):
    """Output must contain at most top_k chunks."""
    chunks = [_make_chunk(f"doc{i}.pdf", f"Text {i}") for i in range(8)]
    scores = [float(i) / 10 for i in range(8)]
    with patch("pipeline.nodes.reranker._get_reranker") as mock_get:
        mock_get.return_value = _mock_reranker(scores)
        from pipeline.nodes.reranker import rerank
        state = {"safe_query": "some query", "chunks": chunks, "top_k": 3}
        result = rerank(state)  # type: ignore[arg-type]
    assert len(result["chunks"]) <= 3


def test_rerank_adds_reranker_score_field(monkeypatch):
    """Each output chunk must have a float 'reranker_score' field."""
    chunks = [_make_chunk("a.pdf", "text A"), _make_chunk("b.pdf", "text B")]
    with patch("pipeline.nodes.reranker._get_reranker") as mock_get:
        mock_get.return_value = _mock_reranker([0.7, 0.4])
        from pipeline.nodes.reranker import rerank
        state = {"safe_query": "some query", "chunks": chunks, "top_k": 5}
        result = rerank(state)  # type: ignore[arg-type]
    for chunk in result["chunks"]:
        assert "reranker_score" in chunk
        assert isinstance(chunk["reranker_score"], float)


def test_rerank_empty_chunks_returns_empty(monkeypatch):
    """Empty chunk input must return empty output without error."""
    from pipeline.nodes.reranker import rerank
    state = {"safe_query": "some query", "chunks": [], "top_k": 5}
    result = rerank(state)  # type: ignore[arg-type]
    assert result["chunks"] == []
