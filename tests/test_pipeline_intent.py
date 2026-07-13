"""
tests/test_pipeline_intent.py — Sprint 5

Unit tests for pipeline/nodes/intent_classifier.py.
All tests mock the Gemini API — no live API calls.
"""

from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock


# ── Helper to mock Gemini response ───────────────────────────────────────────

def _mock_gemini(intent: str):
    """Return a mock Gemini client whose generate_content returns the given intent JSON."""
    mock_resp = MagicMock()
    mock_resp.text = f'{{"intent": "{intent}"}}'
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp
    return mock_client


# ── Import node under test ────────────────────────────────────────────────────

from pipeline.nodes.intent_classifier import classify_intent, _classify


# ── Tests for _classify (unit, mocked LLM) ───────────────────────────────────

def test_simple_query_classified_as_simple(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_A", "fake-key")
    with patch("pipeline.nodes.intent_classifier.genai") as mock_genai:
        mock_genai.Client.return_value = _mock_gemini("simple")
        result = _classify("What is the travel expense reimbursement limit?")
    assert result == "simple"


def test_multi_hop_query_classified_as_multi_hop(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_A", "fake-key")
    with patch("pipeline.nodes.intent_classifier.genai") as mock_genai:
        mock_genai.Client.return_value = _mock_gemini("multi_hop")
        result = _classify("Compare the expense policy and the IT security incident response timeline")
    assert result == "multi_hop"


def test_out_of_scope_classified_correctly(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_A", "fake-key")
    with patch("pipeline.nodes.intent_classifier.genai") as mock_genai:
        mock_genai.Client.return_value = _mock_gemini("out_of_scope")
        result = _classify("What is the GDP of France?")
    assert result == "out_of_scope"


def test_parse_failure_defaults_to_multi_hop(monkeypatch):
    """If Gemini returns unparseable JSON, safe-default is 'multi_hop'."""
    monkeypatch.setenv("GEMINI_API_KEY_A", "fake-key")
    with patch("pipeline.nodes.intent_classifier.genai") as mock_genai:
        bad_resp = MagicMock()
        bad_resp.text = "oops not json"
        mock_genai.Client.return_value.models.generate_content.return_value = bad_resp
        result = _classify("Some query")
    assert result == "multi_hop"


def test_invalid_intent_value_defaults_to_multi_hop(monkeypatch):
    """If Gemini returns an unknown intent string, safe-default is 'multi_hop'."""
    monkeypatch.setenv("GEMINI_API_KEY_A", "fake-key")
    with patch("pipeline.nodes.intent_classifier.genai") as mock_genai:
        mock_genai.Client.return_value = _mock_gemini("unknown_intent")
        result = _classify("Some query")
    assert result == "multi_hop"


def test_empty_query_returns_out_of_scope():
    """Empty query is classified as out_of_scope without an API call."""
    result = _classify("")
    assert result == "out_of_scope"


# ── Tests for classify_intent (full node, mocked LLM + router) ───────────────

def test_classify_intent_returns_required_keys(monkeypatch):
    """Node must return lang, intent, safe_query, pii_flagged."""
    monkeypatch.setenv("GEMINI_API_KEY_A", "fake-key")
    with patch("pipeline.nodes.intent_classifier.genai") as mock_genai:
        mock_genai.Client.return_value = _mock_gemini("simple")
        state = {
            "query": "What is the leave policy?",
            "bu_filter": "hr",
            "top_k": 5,
        }
        result = classify_intent(state)  # type: ignore[arg-type]
    assert "lang" in result
    assert "intent" in result
    assert "safe_query" in result
    assert "pii_flagged" in result


def test_classify_intent_detects_english(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_A", "fake-key")
    with patch("pipeline.nodes.intent_classifier.genai") as mock_genai:
        mock_genai.Client.return_value = _mock_gemini("simple")
        state = {"query": "What is the expense limit?", "bu_filter": "", "top_k": 5}
        result = classify_intent(state)  # type: ignore[arg-type]
    assert result["lang"] == "en"


def test_classify_intent_detects_chinese(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_A", "fake-key")
    with patch("pipeline.nodes.intent_classifier.genai") as mock_genai:
        mock_genai.Client.return_value = _mock_gemini("simple")
        state = {"query": "差旅报销政策是什么？", "bu_filter": "", "top_k": 5}
        result = classify_intent(state)  # type: ignore[arg-type]
    assert result["lang"] == "zh"
