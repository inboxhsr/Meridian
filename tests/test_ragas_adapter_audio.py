"""
tests/test_ragas_adapter_audio.py — Sprint 9 test gate (2 of 4)

Tests for eval/ragas_adapter.py — audio modality surrogate.
All tests are fully mocked; no Gemini API call is made.
"""

import sys
from pathlib import Path

import pytest

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.ragas_adapter import build_ragas_contexts, get_text_surrogate


# ── get_text_surrogate — audio ─────────────────────────────────────────────────

def test_audio_chunk_returns_transcript():
    """Audio chunks must return the Whisper transcript from metadata."""
    chunk = {
        "modality": "audio",
        "chunk_text": "",
        "metadata": {"transcript": "CEO quarterly message transcript text here."},
    }
    result = get_text_surrogate(chunk)
    assert result == "CEO quarterly message transcript text here."


def test_audio_chunk_missing_transcript_returns_empty_string():
    """Audio chunk with no transcript key must return empty string, not raise."""
    chunk = {"modality": "audio", "chunk_text": "", "metadata": {}}
    result = get_text_surrogate(chunk)
    assert result == ""


def test_audio_chunk_none_metadata_returns_empty_string():
    """Audio chunk with metadata=None must not raise."""
    chunk = {"modality": "audio", "chunk_text": "", "metadata": None}
    result = get_text_surrogate(chunk)
    assert result == ""


def test_audio_chunk_empty_transcript_returns_empty_string():
    """Audio chunk with empty transcript string must return empty string."""
    chunk = {"modality": "audio", "chunk_text": "some text", "metadata": {"transcript": ""}}
    result = get_text_surrogate(chunk)
    assert result == ""


# ── get_text_surrogate — text ──────────────────────────────────────────────────

def test_text_chunk_returns_chunk_text():
    """Text chunks must return chunk_text directly."""
    chunk = {"modality": "text", "chunk_text": "Expense limit is $500.", "metadata": {}}
    result = get_text_surrogate(chunk)
    assert result == "Expense limit is $500."


def test_text_chunk_missing_modality_defaults_to_text():
    """If modality key is missing, fallback to text behaviour."""
    chunk = {"chunk_text": "Default text chunk content.", "metadata": {}}
    result = get_text_surrogate(chunk)
    assert result == "Default text chunk content."


def test_text_chunk_empty_chunk_text_returns_empty():
    """Empty chunk_text on a text chunk returns empty string."""
    chunk = {"modality": "text", "chunk_text": "", "metadata": {}}
    result = get_text_surrogate(chunk)
    assert result == ""


# ── build_ragas_contexts ───────────────────────────────────────────────────────

def test_build_ragas_contexts_filters_empty_surrogates():
    """build_ragas_contexts must filter out empty surrogates."""
    hits = [
        {"modality": "text", "chunk_text": "Valid text.", "metadata": {}},
        {"modality": "audio", "chunk_text": "", "metadata": {}},       # empty → filtered
        {"modality": "text", "chunk_text": "Another valid text.", "metadata": {}},
    ]
    contexts = build_ragas_contexts(hits)
    assert len(contexts) == 2
    assert "Valid text." in contexts
    assert "Another valid text." in contexts


def test_build_ragas_contexts_empty_hits_returns_empty_list():
    """Empty hits list must return empty list (not raise)."""
    contexts = build_ragas_contexts([])
    assert contexts == []


def test_build_ragas_contexts_all_audio_with_transcripts():
    """All audio chunks with transcripts must all appear in output."""
    hits = [
        {"modality": "audio", "chunk_text": "", "metadata": {"transcript": "Transcript A."}},
        {"modality": "audio", "chunk_text": "", "metadata": {"transcript": "Transcript B."}},
    ]
    contexts = build_ragas_contexts(hits)
    assert len(contexts) == 2
    assert "Transcript A." in contexts
    assert "Transcript B." in contexts
