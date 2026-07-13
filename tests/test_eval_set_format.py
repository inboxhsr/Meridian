"""
tests/test_eval_set_format.py — Sprint 9 test gate (1 of 4)

Validates the structure of eval/eval_set.json.
Pure file-format test — no live APIs, no Milvus, no model calls.
"""

import json
from pathlib import Path

import pytest

_EVAL_SET_PATH = Path(__file__).parent.parent / "eval" / "eval_set.json"
_REQUIRED_FIELDS = {"id", "question", "ground_truth", "bu_filter", "source_lang", "expected_lang", "unanswerable"}
_VALID_BUS = {"hr", "product", "it_security", "exec_comms", ""}


def _load() -> list[dict]:
    with open(_EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_eval_set_file_exists():
    """eval/eval_set.json must exist on disk."""
    assert _EVAL_SET_PATH.exists(), f"eval_set.json not found at {_EVAL_SET_PATH}"


def test_eval_set_is_valid_json():
    """eval/eval_set.json must parse as valid JSON."""
    data = _load()
    assert isinstance(data, list)


def test_eval_set_has_at_least_80_entries():
    """Must contain at least 80 QA pairs."""
    data = _load()
    assert len(data) >= 80, f"Expected ≥80 entries, got {len(data)}"


def test_all_entries_have_required_fields():
    """Every entry must have all required fields."""
    data = _load()
    for entry in data:
        missing = _REQUIRED_FIELDS - set(entry.keys())
        assert not missing, f"Entry '{entry.get('id', '?')}' is missing fields: {missing}"


def test_all_ids_are_unique():
    """No two entries may share the same ID."""
    data = _load()
    ids = [e["id"] for e in data]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"Duplicate IDs found: {duplicates}"


def test_all_ids_are_non_empty_strings():
    """All IDs must be non-empty strings."""
    data = _load()
    for entry in data:
        assert isinstance(entry["id"], str) and entry["id"], \
            f"Entry has empty or non-string id: {entry}"


def test_all_questions_are_non_empty():
    """All questions must be non-empty strings."""
    data = _load()
    for entry in data:
        assert isinstance(entry["question"], str) and entry["question"].strip(), \
            f"Entry '{entry['id']}' has empty question"


def test_all_ground_truths_are_non_empty():
    """All ground_truth values must be non-empty strings."""
    data = _load()
    for entry in data:
        assert isinstance(entry["ground_truth"], str) and entry["ground_truth"].strip(), \
            f"Entry '{entry['id']}' has empty ground_truth"


def test_bu_filter_values_are_strings():
    """bu_filter must be a string (empty string is allowed for cross-BU queries)."""
    data = _load()
    for entry in data:
        assert isinstance(entry["bu_filter"], str), \
            f"Entry '{entry['id']}' has non-string bu_filter: {entry['bu_filter']!r}"


def test_unanswerable_flag_is_boolean():
    """unanswerable must be a boolean."""
    data = _load()
    for entry in data:
        assert isinstance(entry["unanswerable"], bool), \
            f"Entry '{entry['id']}' has non-bool unanswerable: {entry['unanswerable']!r}"


def test_source_lang_values_are_valid():
    """source_lang must be one of: en, zh, hi."""
    valid = {"en", "zh", "hi"}
    data = _load()
    for entry in data:
        assert entry["source_lang"] in valid, \
            f"Entry '{entry['id']}' has invalid source_lang: '{entry['source_lang']}'"


def test_expected_lang_values_are_valid():
    """expected_lang must be one of: en, zh, hi."""
    valid = {"en", "zh", "hi"}
    data = _load()
    for entry in data:
        assert entry["expected_lang"] in valid, \
            f"Entry '{entry['id']}' has invalid expected_lang: '{entry['expected_lang']}'"


def test_cross_lingual_subset_has_at_least_25_entries():
    """Must have at least 25 cross-lingual gotcha pairs (source_lang != expected_lang)."""
    data = _load()
    cross = [e for e in data if e["source_lang"] != e["expected_lang"]]
    assert len(cross) >= 25, f"Expected ≥25 cross-lingual entries, got {len(cross)}"


def test_unanswerable_subset_has_at_least_5_entries():
    """Must have at least 5 unanswerable pairs for abstention testing."""
    data = _load()
    unans = [e for e in data if e.get("unanswerable")]
    assert len(unans) >= 5, f"Expected ≥5 unanswerable entries, got {len(unans)}"


def test_multiple_languages_present():
    """Eval set must include questions in English, Chinese, and Hindi."""
    data = _load()
    source_langs = {e["source_lang"] for e in data}
    for lang in ("en", "zh", "hi"):
        assert lang in source_langs, f"Missing source_lang '{lang}' in eval set"


def test_multiple_bus_represented():
    """Eval set must include at least hr, product, and it_security BU filters."""
    data = _load()
    bus = {e["bu_filter"] for e in data if e["bu_filter"]}
    for bu in ("hr", "product", "it_security"):
        assert bu in bus, f"Missing BU '{bu}' in eval set"
