"""
tests/test_ragas_adapter_image.py — Sprint 9 test gate (3 of 4)

Tests for eval/ragas_adapter.py — image modality surrogate.
All Gemini API calls are mocked; no live API access required.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import eval.ragas_adapter as adapter
from eval.ragas_adapter import build_ragas_contexts, get_text_surrogate


# ── get_text_surrogate — image ─────────────────────────────────────────────────

def test_image_chunk_calls_caption_and_returns_result():
    """Image chunks must call _caption_image and return its result."""
    chunk = {
        "modality": "image",
        "chunk_text": "",
        "metadata": {},
        "doc_id": "product_demo_deck.en_slide001",
    }
    with patch.object(adapter, "_caption_image", return_value="Slide showing Q3 revenue chart.") as mock_cap:
        result = get_text_surrogate(chunk)
    mock_cap.assert_called_once_with("product_demo_deck.en_slide001")
    assert result == "Slide showing Q3 revenue chart."


def test_image_chunk_uses_source_file_as_doc_id_fallback():
    """If doc_id is absent, source_file must be used as fallback for captioning."""
    chunk = {
        "modality": "image",
        "chunk_text": "",
        "metadata": {},
        "source_file": "exec_comms_annual_review_2025.en_slide003",
    }
    with patch.object(adapter, "_caption_image", return_value="Annual review slide.") as mock_cap:
        result = get_text_surrogate(chunk)
    mock_cap.assert_called_once_with("exec_comms_annual_review_2025.en_slide003")
    assert result == "Annual review slide."


def test_image_chunk_caption_failure_returns_empty_string():
    """If _caption_image raises, get_text_surrogate must return '' without raising."""
    chunk = {
        "modality": "image",
        "chunk_text": "",
        "metadata": {},
        "doc_id": "bad_slide",
    }
    with patch.object(adapter, "_caption_image", side_effect=Exception("API error")):
        result = get_text_surrogate(chunk)
    assert result == ""


def test_image_chunk_empty_caption_returns_empty_string():
    """If _caption_image returns empty string, surrogate must also be empty."""
    chunk = {"modality": "image", "chunk_text": "", "metadata": {}, "doc_id": "slide_x"}
    with patch.object(adapter, "_caption_image", return_value=""):
        result = get_text_surrogate(chunk)
    assert result == ""


def test_image_chunk_caption_returns_non_empty_string():
    """Caption result must be a non-empty string when Gemini responds."""
    chunk = {"modality": "image", "chunk_text": "", "metadata": {}, "doc_id": "product_demo_deck.en_slide002"}
    caption = "Product demo slide showing feature comparison table."
    with patch.object(adapter, "_caption_image", return_value=caption):
        result = get_text_surrogate(chunk)
    assert isinstance(result, str)
    assert len(result) > 0


# ── build_ragas_contexts — mixed modalities ────────────────────────────────────

def test_build_ragas_contexts_mixed_modalities():
    """
    build_ragas_contexts must handle text, audio, and image chunks together.
    Image caption is mocked; audio transcript read from metadata.
    """
    hits = [
        {"modality": "text", "chunk_text": "Expense limit is $500.", "metadata": {}},
        {
            "modality": "audio",
            "chunk_text": "",
            "metadata": {"transcript": "CEO quarterly message."},
        },
        {
            "modality": "image",
            "chunk_text": "",
            "metadata": {},
            "doc_id": "exec_comms_q3_allhands.en_slide001",
        },
    ]
    with patch.object(adapter, "_caption_image", return_value="Q3 all-hands slide with agenda."):
        contexts = build_ragas_contexts(hits)

    assert len(contexts) == 3
    assert "Expense limit is $500." in contexts
    assert "CEO quarterly message." in contexts
    assert "Q3 all-hands slide with agenda." in contexts


def test_build_ragas_contexts_image_caption_failure_excluded():
    """
    If image caption fails for one chunk, that chunk is excluded from contexts
    but other chunks still appear.
    """
    hits = [
        {"modality": "text", "chunk_text": "Valid text chunk.", "metadata": {}},
        {"modality": "image", "chunk_text": "", "metadata": {}, "doc_id": "broken_slide"},
    ]
    with patch.object(adapter, "_caption_image", side_effect=Exception("Gemini down")):
        contexts = build_ragas_contexts(hits)

    assert len(contexts) == 1
    assert "Valid text chunk." in contexts
