"""
tests/test_reranker_ordering.py — Sprint 6

Tests that the BGE reranker actually reorders chunks by relevance.
These tests use the REAL CrossEncoder model (uses cached download from HuggingFace).

Mark with pytest.mark.slow — skip in CI with: pytest -m "not slow"
"""

from __future__ import annotations
import math
import pytest


pytestmark = pytest.mark.slow


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@pytest.fixture(scope="module")
def real_reranker():
    """Load the real BGE reranker once for this test module."""
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    except Exception as e:
        pytest.skip(f"sentence-transformers not installed or model unavailable: {e}")


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


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_relevant_chunk_ranks_above_irrelevant(real_reranker):
    """A highly relevant chunk must score higher than a clearly irrelevant one."""
    query = "What is the travel expense reimbursement limit?"
    relevant_text = (
        "The travel expense reimbursement limit is $500 per trip for economy class flights. "
        "Business class is approved for flights over 8 hours."
    )
    irrelevant_text = (
        "All employees must complete the annual cybersecurity awareness training by December 31. "
        "Failure to complete will result in system access suspension."
    )

    pairs = [[query, relevant_text], [query, irrelevant_text]]
    raw = real_reranker.predict(pairs)
    scores = [_sigmoid(float(s)) for s in raw]

    assert scores[0] > scores[1], (
        f"Expected relevant chunk (score={scores[0]:.3f}) to outscore "
        f"irrelevant chunk (score={scores[1]:.3f})"
    )


def test_output_sorted_descending_by_score(real_reranker):
    """Output chunks must be sorted in descending order of reranker_score."""
    from unittest.mock import patch

    query = "What is the expense reimbursement policy?"
    chunks = [
        _make_chunk("it_security.pdf", "Employees must use VPN for remote access to corporate systems."),
        _make_chunk("hr_expense.pdf",  "Travel expenses are reimbursed up to $500 per business trip."),
        _make_chunk("product_spec.pdf","Meridian 3.0 supports 50 concurrent API integrations per tenant."),
    ]

    with patch("pipeline.nodes.reranker._get_reranker", return_value=real_reranker):
        from pipeline.nodes.reranker import rerank
        state = {"safe_query": query, "chunks": chunks, "top_k": 10}
        result = rerank(state)  # type: ignore[arg-type]

    output_scores = [c["reranker_score"] for c in result["chunks"]]
    assert output_scores == sorted(output_scores, reverse=True), (
        f"Chunks not sorted descending: {output_scores}"
    )


def test_reranker_changes_retrieval_order(real_reranker):
    """Reranker must promote the most relevant chunk to top-1."""
    from unittest.mock import patch

    query = "How do I submit a travel expense report?"
    # ANN retrieval order: IT doc first, then HR doc (simulating wrong ANN ranking)
    chunks = [
        _make_chunk("it_security.pdf",  "Employees must report security incidents within 24 hours.", score=0.91),
        _make_chunk("hr_expense.pdf",   "Submit travel expense reports via the HR portal within 30 days.", score=0.88),
        _make_chunk("product_spec.pdf", "API rate limit is 1000 requests per minute per tenant.", score=0.75),
    ]
    original_top1 = chunks[0]["source_file"]  # it_security.pdf (wrong ANN top-1)

    with patch("pipeline.nodes.reranker._get_reranker", return_value=real_reranker):
        from pipeline.nodes.reranker import rerank
        state = {"safe_query": query, "chunks": chunks, "top_k": 10}
        result = rerank(state)  # type: ignore[arg-type]

    reranked_top1 = result["chunks"][0]["source_file"]

    # hr_expense.pdf should now rank first (most relevant to expense report query)
    assert reranked_top1 == "hr_expense.pdf", (
        f"Expected hr_expense.pdf as top-1 after reranking, got {reranked_top1}"
    )
    # And the order must differ from the original ANN order
    assert reranked_top1 != original_top1, "Reranker produced same top-1 as original retrieval"
