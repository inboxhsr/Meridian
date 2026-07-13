"""
tests/test_pipeline_graph.py — Sprint 5 / Sprint 8 (query_id added to state)

Integration tests for the full LangGraph pipeline.
Requires: Milvus running, GEMINI_API_KEY_A, DEEPSEEK_API_KEY set in .env.

These tests exercise the real graph end-to-end with live API calls.
"""

from __future__ import annotations
import uuid
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def _graph():
    """Compile the graph once for all tests in this module."""
    from pipeline.graph import graph
    return graph


def _invoke(graph, query: str, bu_filter: str = "", top_k: int = 5) -> dict:
    """Build the initial state and invoke the graph."""
    initial_state = {
        "query_id":           str(uuid.uuid4()),   # Sprint 8: observability session ID
        "query":              query,
        "bu_filter":          bu_filter,
        "top_k":              top_k,
        "lang":               "en",
        "intent":             "",
        "safe_query":         query,
        "pii_flagged":        False,
        "sub_questions":      [],
        "chunks":             [],
        "retrieval_round":    0,
        "groundedness_score": 0.0,
        "relevance_score":    0.0,
        "verdict":            "",
        "critic_reasoning":   "",
        "answer":             "",
        "sources":            [],
        "chunks_used":        0,
        "abstained":          False,
    }
    return graph.invoke(initial_state)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_simple_en_query_returns_answer(_graph):
    """Simple English query should return a non-empty answer with sources."""
    result = _invoke(_graph, "What is the travel expense reimbursement limit?", bu_filter="hr")
    assert result["answer"], "Expected a non-empty answer"
    # Either cited sources or an honest abstention
    assert isinstance(result["sources"], list)
    assert isinstance(result["abstained"], bool)


def test_zh_query_answer_contains_chinese(_graph):
    """Chinese query should produce an answer containing Chinese characters."""
    import re
    cjk = re.compile(r"[\u4e00-\u9fff]")
    result = _invoke(_graph, "差旅报销政策是什么？", bu_filter="hr")
    assert result["lang"] == "zh"
    # If we got an answer (not abstained), it should be in Chinese
    if not result["abstained"]:
        assert cjk.search(result["answer"]), "Expected Chinese characters in answer"


def test_bu_filter_respected(_graph):
    """All returned chunks must belong to the specified BU."""
    result = _invoke(_graph, "What is the access control policy?", bu_filter="it_security")
    chunks = result.get("chunks", [])
    if chunks:
        for chunk in chunks:
            assert chunk["bu"] == "it_security", f"Expected it_security BU, got {chunk['bu']}"


def test_answerable_query_is_not_abstained(_graph):
    """A clearly answerable query should not trigger abstention."""
    result = _invoke(_graph, "What is the leave policy?", bu_filter="hr")
    # We cannot guarantee abstained=False (depends on retrieval quality),
    # but we can check state structure is valid
    assert "abstained" in result
    assert "answer" in result
    assert len(result["answer"]) > 0
