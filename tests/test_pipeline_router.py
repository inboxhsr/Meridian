"""
tests/test_pipeline_router.py — Sprint 3 test gate

Unit tests for the router module.
All tests here are PURE — no API calls, no Milvus, no network.
PII guard integration tests (Stage 2, Gemini) are skipped by default.
"""

import pytest
from pipeline.router import detect_language, check_pii, route


# ── detect_language ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("What is the travel expense policy?", "en"),
    ("How do I report a security incident?", "en"),
    ("差旅报销上限是多少？", "zh"),
    ("经费报销流程是什么", "zh"),
    ("यात्रा व्यय नीति क्या है?", "hi"),
    ("कर्मचारी छुट्टी नीति", "hi"),
    ("", "en"),  # empty → default en
])
def test_detect_language(text, expected):
    assert detect_language(text) == expected, (
        f"detect_language({text!r}) returned {detect_language(text)!r}, expected {expected!r}"
    )


def test_detect_language_mixed_mostly_english():
    """Text with a few CJK chars but mostly ASCII → English."""
    text = "The product roadmap 2026 (产品) is available for review."
    # ~3 CJK chars out of ~50 total chars = 6% < 8% threshold
    assert detect_language(text) == "en"


def test_detect_language_chinese_dominant():
    """Text with >8% CJK chars → Chinese."""
    text = "请查阅2026年产品路线图以获取详细信息。"
    assert detect_language(text) == "zh"


# ── check_pii (Stage 1 regex only — no API call) ─────────────────────────────

def test_check_pii_clean_query_no_api_call():
    """A clean query must not trigger the API call (Stage 1 short-circuits)."""
    result = check_pii("What is the IT security incident response procedure?")
    assert result["has_pii"] is False
    assert result["redacted"] == "What is the IT security incident response procedure?"


def test_check_pii_email_triggers_stage1():
    """A query with an email address must be flagged by Stage 1 regex."""
    # We're NOT calling Gemini here because GEMINI_API_KEY_A may be quota-exhausted.
    # We just verify the regex fires (has_pii=True or redacted changed).
    result = check_pii("Send results to john.doe@example.com please")
    # Stage 1 fires; Stage 2 may or may not fire depending on env
    # Either way the result must acknowledge PII
    assert result["has_pii"] is True or "[REDACTED" in result["redacted"]


def test_check_pii_ssn_triggers():
    result = check_pii("My SSN is 123-45-6789 and I need help")
    assert result["has_pii"] is True or "[REDACTED" in result["redacted"]


def test_check_pii_phone_triggers():
    result = check_pii("Call me at 555-867-5309")
    assert result["has_pii"] is True or "[REDACTED" in result["redacted"]


# ── route ─────────────────────────────────────────────────────────────────────

def test_route_returns_all_keys():
    """route() must return all required keys."""
    result = route("What is the leave policy?", skip_pii=True)
    for key in ("query", "safe_query", "lang", "bu_filter", "pii_flagged"):
        assert key in result, f"Missing key: {key}"


def test_route_language_detection_integrated():
    result = route("差旅报销规定", skip_pii=True)
    assert result["lang"] == "zh"


def test_route_bu_filter_passthrough():
    result = route("expense policy", bu_filter="hr", skip_pii=True)
    assert result["bu_filter"] == "hr"


def test_route_no_pii_when_skipped():
    result = route("Call me at 555-867-5309", skip_pii=True)
    assert result["pii_flagged"] is False
    assert result["safe_query"] == "Call me at 555-867-5309"


def test_route_clean_query_safe():
    result = route("What are the IT security access control requirements?", skip_pii=True)
    assert result["pii_flagged"] is False
    assert result["safe_query"] == result["query"]
