"""
tests/test_pipeline_rewriter.py — Sprint 5

Unit tests for pipeline/nodes/query_rewriter.py.
All tests mock the DeepSeek API — no live API calls.
"""

from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock

from pipeline.nodes.query_rewriter import rewrite_query, _decompose


# ── Helper ────────────────────────────────────────────────────────────────────

def _mock_deepseek(questions: list[str]):
    """Return a mock OpenAI client whose completions return the given list."""
    import json
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps(questions)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    return mock_client


# ── Tests for _decompose ──────────────────────────────────────────────────────

def test_compound_query_returns_multiple_subquestions(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    with patch("pipeline.nodes.query_rewriter.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_deepseek([
            "What is the expense reimbursement limit?",
            "What is the IT security incident response timeline?",
        ])
        result = _decompose("Compare the expense policy and the IT security incident response timeline")
    assert len(result) > 1
    assert all(isinstance(q, str) for q in result)


def test_simple_query_returns_exactly_one(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    with patch("pipeline.nodes.query_rewriter.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_deepseek(["What is the travel expense limit?"])
        result = _decompose("What is the travel expense limit?")
    assert len(result) == 1


def test_subquestions_are_nonempty_strings(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    with patch("pipeline.nodes.query_rewriter.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_deepseek(["Q1?", "Q2?", "Q3?"])
        result = _decompose("Some compound query")
    assert all(q and q.strip() for q in result)


def test_parse_failure_returns_original_query(monkeypatch):
    """If DeepSeek returns unparseable JSON, fall back to [original_query]."""
    original = "What is the IT access policy?"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    with patch("pipeline.nodes.query_rewriter.OpenAI") as MockOpenAI:
        bad_resp = MagicMock()
        bad_resp.choices[0].message.content = "not valid json"
        MockOpenAI.return_value.chat.completions.create.return_value = bad_resp
        result = _decompose(original)
    assert result == [original]


def test_chinese_query_stays_in_chinese(monkeypatch):
    """Rewriter must not translate — Chinese query should produce Chinese sub-questions."""
    zh_query = "差旅报销政策和IT安全政策有什么区别？"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    with patch("pipeline.nodes.query_rewriter.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_deepseek(["差旅报销政策是什么？", "IT安全政策是什么？"])
        result = _decompose(zh_query)
    # At least one sub-question should contain Chinese characters
    import re
    cjk = re.compile(r"[\u4e00-\u9fff]")
    assert any(cjk.search(q) for q in result)


# ── Tests for rewrite_query (node interface) ──────────────────────────────────

def test_rewrite_query_node_sets_sub_questions(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    with patch("pipeline.nodes.query_rewriter.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_deepseek(["Q1?", "Q2?"])
        state = {"safe_query": "Tell me about Q1 and Q2"}
        result = rewrite_query(state)  # type: ignore[arg-type]
    assert "sub_questions" in result
    assert isinstance(result["sub_questions"], list)
    assert len(result["sub_questions"]) >= 1
