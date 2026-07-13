"""
tests/test_pipeline_abstain.py — Sprint 5

Tests for pipeline/nodes/abstainer.py and graph abstention path.
These are pure unit tests — no API calls or Milvus required.
"""

from __future__ import annotations
import pytest

from pipeline.nodes.abstainer import abstain


# ── Tests for abstain node ────────────────────────────────────────────────────

def test_out_of_scope_sets_abstained_true():
    """out_of_scope intent → abstained == True."""
    state = {
        "intent": "out_of_scope",
        "lang": "en",
        "retrieval_round": 0,
    }
    result = abstain(state)  # type: ignore[arg-type]
    assert result["abstained"] is True


def test_abstain_answer_is_nonempty():
    """Abstainer must always return a non-empty answer string."""
    state = {"intent": "simple", "lang": "en", "retrieval_round": 3}
    result = abstain(state)  # type: ignore[arg-type]
    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0


def test_abstain_sources_are_empty():
    """Abstainer should return no sources."""
    state = {"intent": "simple", "lang": "en", "retrieval_round": 3}
    result = abstain(state)  # type: ignore[arg-type]
    assert result["sources"] == []
    assert result["chunks_used"] == 0


def test_abstain_chinese_returns_chinese_message():
    """Chinese language → Chinese abstain message."""
    state = {"intent": "simple", "lang": "zh", "retrieval_round": 3}
    result = abstain(state)  # type: ignore[arg-type]
    import re
    cjk = re.compile(r"[\u4e00-\u9fff]")
    assert cjk.search(result["answer"]), "Expected Chinese characters in abstain message"


def test_out_of_scope_message_differs_from_retry_exhausted_message():
    """out_of_scope and retry-exhausted paths must produce distinct messages."""
    oos_result = abstain({"intent": "out_of_scope", "lang": "en", "retrieval_round": 0})  # type: ignore[arg-type]
    exhausted_result = abstain({"intent": "simple", "lang": "en", "retrieval_round": 3})  # type: ignore[arg-type]
    assert oos_result["answer"] != exhausted_result["answer"]
