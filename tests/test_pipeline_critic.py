"""
tests/test_pipeline_critic.py — Sprint 5

Unit tests for pipeline/nodes/critic.py.
All tests mock the DeepSeek API — no live API calls.
"""

from __future__ import annotations
import json
import pytest
from unittest.mock import patch, MagicMock

from pipeline.nodes.critic import grade_context, _grade, _GROUNDEDNESS_THRESHOLD, _RELEVANCE_THRESHOLD, _MAX_ROUNDS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_deepseek(groundedness: float, relevance: float, reasoning: str = "test"):
    resp_body = json.dumps({
        "groundedness_score": groundedness,
        "relevance_score": relevance,
        "verdict": "sufficient",  # node overrides this, so value here doesn't matter
        "reasoning": reasoning,
    })
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = resp_body
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    return mock_client


def _make_chunk(text: str = "Some context text") -> dict:
    return {"source_file": "hr_policy.en.pdf", "text": text, "bu": "hr",
            "lang": "en", "modality": "text", "score": 0.8, "chunk_index": 0}


# ── Verdict logic tests ───────────────────────────────────────────────────────

def test_strong_context_returns_sufficient(monkeypatch):
    """groundedness >= 0.7 AND relevance >= 0.6 → 'sufficient'."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    with patch("pipeline.nodes.critic.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_deepseek(groundedness=0.85, relevance=0.75)
        state = {
            "safe_query": "What is the expense limit?",
            "chunks": [_make_chunk()],
            "retrieval_round": 1,
        }
        result = grade_context(state)  # type: ignore[arg-type]
    assert result["verdict"] == "sufficient"


def test_weak_context_round_1_returns_retry(monkeypatch):
    """Below threshold AND round < MAX_ROUNDS → 'retry'."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    with patch("pipeline.nodes.critic.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_deepseek(groundedness=0.3, relevance=0.2)
        state = {
            "safe_query": "What is the expense limit?",
            "chunks": [_make_chunk()],
            "retrieval_round": 1,
        }
        result = grade_context(state)  # type: ignore[arg-type]
    assert result["verdict"] == "retry"


def test_weak_context_max_rounds_returns_abstain(monkeypatch):
    """Below threshold AND round >= MAX_ROUNDS → 'abstain'."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    with patch("pipeline.nodes.critic.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_deepseek(groundedness=0.3, relevance=0.2)
        state = {
            "safe_query": "What is the expense limit?",
            "chunks": [_make_chunk()],
            "retrieval_round": _MAX_ROUNDS,
        }
        result = grade_context(state)  # type: ignore[arg-type]
    assert result["verdict"] == "abstain"


def test_parse_failure_returns_conservative_retry(monkeypatch):
    """If DeepSeek returns bad JSON, safe-default is retry with 0.0 scores."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    with patch("pipeline.nodes.critic.OpenAI") as MockOpenAI:
        bad_resp = MagicMock()
        bad_resp.choices[0].message.content = "not valid json"
        MockOpenAI.return_value.chat.completions.create.return_value = bad_resp
        state = {
            "safe_query": "What is the expense limit?",
            "chunks": [_make_chunk()],
            "retrieval_round": 1,
        }
        result = grade_context(state)  # type: ignore[arg-type]
    assert result["verdict"] == "retry"
    assert result["groundedness_score"] == 0.0


def test_scores_are_floats_in_range(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    with patch("pipeline.nodes.critic.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_deepseek(groundedness=0.75, relevance=0.65)
        state = {
            "safe_query": "Some question",
            "chunks": [_make_chunk()],
            "retrieval_round": 1,
        }
        result = grade_context(state)  # type: ignore[arg-type]
    assert isinstance(result["groundedness_score"], float)
    assert isinstance(result["relevance_score"], float)
    assert 0.0 <= result["groundedness_score"] <= 1.0
    assert 0.0 <= result["relevance_score"] <= 1.0


def test_critic_reasoning_is_nonempty_string(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    with patch("pipeline.nodes.critic.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_deepseek(0.8, 0.7, reasoning="Evidence is clear and on-topic.")
        state = {
            "safe_query": "Some question",
            "chunks": [_make_chunk()],
            "retrieval_round": 1,
        }
        result = grade_context(state)  # type: ignore[arg-type]
    assert isinstance(result["critic_reasoning"], str)
    assert len(result["critic_reasoning"]) > 0
