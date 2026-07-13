"""
tests/test_run_eval.py — Sprint 9 test gate (4 of 4)

Tests for eval/run_eval.py — evaluation runner.

All tests mock _call_api and _ragas_evaluate so no live API calls or
Milvus connection is required. This is a pure unit test file.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import eval.run_eval as run_eval_mod
from eval.run_eval import run_eval_on_samples

# ── Shared fixtures ────────────────────────────────────────────────────────────

_MOCK_SAMPLES = [
    {
        "id": "en_hr_001",
        "question": "What is the expense reimbursement limit?",
        "ground_truth": "The maximum is $500 per trip.",
        "bu_filter": "hr",
        "source_lang": "en",
        "expected_lang": "en",
        "unanswerable": False,
    },
    {
        "id": "en_it_001",
        "question": "What is the password minimum length?",
        "ground_truth": "Twelve characters minimum.",
        "bu_filter": "it_security",
        "source_lang": "en",
        "expected_lang": "en",
        "unanswerable": False,
    },
    {
        "id": "hi_en_001",
        "question": "पासवर्ड की न्यूनतम लंबाई क्या है?",
        "ground_truth": "न्यूनतम 12 अक्षर।",
        "bu_filter": "it_security",
        "source_lang": "en",
        "expected_lang": "hi",
        "unanswerable": False,
    },
    {
        "id": "unans_001",
        "question": "What is the CEO personal home address?",
        "ground_truth": "This information is not available.",
        "bu_filter": "",
        "source_lang": "en",
        "expected_lang": "en",
        "unanswerable": True,
    },
    {
        "id": "unans_002",
        "question": "What is Meridian stock price today?",
        "ground_truth": "This information is not available.",
        "bu_filter": "",
        "source_lang": "en",
        "expected_lang": "en",
        "unanswerable": True,
    },
]

_MOCK_API_RESPONSE_ANSWERED = {
    "answer": "The expense reimbursement limit is $500 per trip.",
    "hits": [
        {"modality": "text", "chunk_text": "Expense limit $500.", "metadata": {}, "source_file": "hr_expense_policy.en.pdf"},
    ],
}

_MOCK_API_RESPONSE_ABSTAINED = {
    "answer": "Based on available searches, insufficient grounded evidence was found to answer this question.",
    "hits": [],
}


# ── Test: output structure ─────────────────────────────────────────────────────

def test_run_eval_output_has_required_top_level_keys():
    """run_eval_on_samples must return a dict with all required top-level keys."""
    with patch.object(run_eval_mod, "_call_api", return_value=_MOCK_API_RESPONSE_ANSWERED):
        with patch.object(run_eval_mod, "_ragas_evaluate", return_value={}):
            result = run_eval_on_samples(_MOCK_SAMPLES[:2], api_url="http://localhost:8000", sleep_between=0)
    for key in ("run_timestamp", "n_samples", "metrics", "custom"):
        assert key in result, f"Missing key '{key}' in result"


def test_run_eval_output_custom_has_required_keys():
    """result['custom'] must contain all expected sub-keys."""
    with patch.object(run_eval_mod, "_call_api", return_value=_MOCK_API_RESPONSE_ANSWERED):
        with patch.object(run_eval_mod, "_ragas_evaluate", return_value={}):
            result = run_eval_on_samples(_MOCK_SAMPLES[:2], api_url="http://localhost:8000", sleep_between=0)
    custom = result["custom"]
    for key in ("cross_lingual_accuracy", "cross_lingual_total", "abstention_rate", "unanswerable_total"):
        assert key in custom, f"Missing custom key '{key}'"


def test_run_eval_n_samples_matches_input():
    """n_samples in result must equal the number of entries passed in."""
    with patch.object(run_eval_mod, "_call_api", return_value=_MOCK_API_RESPONSE_ANSWERED):
        with patch.object(run_eval_mod, "_ragas_evaluate", return_value={}):
            result = run_eval_on_samples(_MOCK_SAMPLES, api_url="http://localhost:8000", sleep_between=0)
    assert result["n_samples"] == len(_MOCK_SAMPLES)


def test_run_eval_completes_without_error():
    """run_eval_on_samples must complete without raising on valid input."""
    with patch.object(run_eval_mod, "_call_api", return_value=_MOCK_API_RESPONSE_ANSWERED):
        with patch.object(run_eval_mod, "_ragas_evaluate", return_value={"context_precision": 0.9}):
            result = run_eval_on_samples(_MOCK_SAMPLES, api_url="http://localhost:8000", sleep_between=0)
    assert result is not None


# ── Test: cross-lingual accuracy ───────────────────────────────────────────────

def test_cross_lingual_accuracy_counted_correctly():
    """Cross-lingual pairs that receive a non-abstained answer are counted."""
    # _MOCK_SAMPLES has 1 cross-lingual entry (hi_en_001, source_lang=en, expected_lang=hi)
    with patch.object(run_eval_mod, "_call_api", return_value=_MOCK_API_RESPONSE_ANSWERED):
        with patch.object(run_eval_mod, "_ragas_evaluate", return_value={}):
            result = run_eval_on_samples(_MOCK_SAMPLES, api_url="http://localhost:8000", sleep_between=0)
    assert result["custom"]["cross_lingual_total"] == 1
    assert result["custom"]["cross_lingual_answered"] == 1
    assert result["custom"]["cross_lingual_accuracy"] == 1.0


def test_cross_lingual_accuracy_zero_when_all_abstained():
    """If cross-lingual queries all return abstention signal, accuracy = 0."""
    with patch.object(run_eval_mod, "_call_api", return_value=_MOCK_API_RESPONSE_ABSTAINED):
        with patch.object(run_eval_mod, "_ragas_evaluate", return_value={}):
            result = run_eval_on_samples(_MOCK_SAMPLES, api_url="http://localhost:8000", sleep_between=0)
    assert result["custom"]["cross_lingual_accuracy"] == 0.0


# ── Test: abstention rate ──────────────────────────────────────────────────────

def test_abstention_rate_correct_when_all_unanswerable_abstained():
    """Abstention rate = 1.0 when all unanswerable entries return abstention signal."""
    with patch.object(run_eval_mod, "_call_api", return_value=_MOCK_API_RESPONSE_ABSTAINED):
        with patch.object(run_eval_mod, "_ragas_evaluate", return_value={}):
            result = run_eval_on_samples(_MOCK_SAMPLES, api_url="http://localhost:8000", sleep_between=0)
    assert result["custom"]["unanswerable_total"] == 2
    assert result["custom"]["abstention_rate"] == 1.0


def test_abstention_rate_zero_when_none_abstained():
    """Abstention rate = 0.0 when unanswerable entries receive confident (wrong) answers."""
    with patch.object(run_eval_mod, "_call_api", return_value=_MOCK_API_RESPONSE_ANSWERED):
        with patch.object(run_eval_mod, "_ragas_evaluate", return_value={}):
            result = run_eval_on_samples(_MOCK_SAMPLES, api_url="http://localhost:8000", sleep_between=0)
    assert result["custom"]["abstention_rate"] == 0.0


# ── Test: API failure handling ─────────────────────────────────────────────────

def test_run_eval_handles_api_failure_gracefully():
    """If _call_api returns empty answer, run_eval must still complete without raising."""
    with patch.object(run_eval_mod, "_call_api", return_value={"answer": "", "hits": []}):
        with patch.object(run_eval_mod, "_ragas_evaluate", return_value={}):
            result = run_eval_on_samples(_MOCK_SAMPLES[:2], api_url="http://localhost:8000", sleep_between=0)
    assert result is not None
    assert result["n_samples"] == 2


def test_run_eval_handles_ragas_failure_gracefully():
    """If _ragas_evaluate raises, run_eval must log the error and return empty metrics dict."""
    with patch.object(run_eval_mod, "_call_api", return_value=_MOCK_API_RESPONSE_ANSWERED):
        with patch.object(run_eval_mod, "_ragas_evaluate", side_effect=Exception("RAGAS crashed")):
            result = run_eval_on_samples(_MOCK_SAMPLES[:2], api_url="http://localhost:8000", sleep_between=0)
    assert result["metrics"] == {}


# ── Test: RAGAS metrics forwarded ──────────────────────────────────────────────

def test_run_eval_forwards_ragas_metrics():
    """Metrics returned by _ragas_evaluate must appear in result['metrics']."""
    mock_metrics = {"context_precision": 0.85, "faithfulness": 0.91}
    with patch.object(run_eval_mod, "_call_api", return_value=_MOCK_API_RESPONSE_ANSWERED):
        with patch.object(run_eval_mod, "_ragas_evaluate", return_value=mock_metrics):
            result = run_eval_on_samples(_MOCK_SAMPLES[:2], api_url="http://localhost:8000", sleep_between=0)
    assert result["metrics"]["context_precision"] == pytest.approx(0.85)
    assert result["metrics"]["faithfulness"] == pytest.approx(0.91)
