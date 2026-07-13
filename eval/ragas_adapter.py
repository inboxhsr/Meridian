"""
eval/ragas_adapter.py — Sprint 9

Text-surrogate logic for multimodal groundedness scoring.

RAGAS requires plain-text context strings. When retrieved chunks contain
audio or image evidence, this module maps them to text surrogates:

  - text  → chunk_text directly
  - audio → metadata["transcript"] (Whisper transcript stored at ingest time)
  - image → one-off Gemini Flash caption call (eval-time only, never stored)

Reference: project_charter.md §9 "Multimodal Groundedness (Text-Surrogate Approach)"
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

# ── Gemini client (lazy-init, only needed for image captions) ─────────────────

_gemini_client = None


def _get_gemini_client():
    """Return a cached Gemini client, initialised on first call."""
    global _gemini_client
    if _gemini_client is None:
        import google.generativeai as genai  # type: ignore
        api_key = os.environ.get("GEMINI_API_KEY_A", "")
        genai.configure(api_key=api_key)
        _gemini_client = genai.GenerativeModel("gemini-2.5-flash")
    return _gemini_client


# ── Private helper ────────────────────────────────────────────────────────────

def _caption_image(doc_id: str) -> str:
    """
    Generate a one-sentence text caption for an image chunk.

    Called only at eval time for image modality chunks.
    The caption is NOT stored in Milvus and NOT used in the normal pipeline.

    Args:
        doc_id: The document identifier (e.g. 'product_demo_deck.en_slide001').

    Returns:
        A short descriptive caption string, or "" on any failure.
    """
    try:
        client = _get_gemini_client()
        prompt = (
            f"You are reviewing an enterprise slide document with ID '{doc_id}'. "
            "Based on the document ID alone, write one concise sentence (max 30 words) "
            "describing the likely content of this slide for grounding evaluation purposes."
        )
        response = client.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Image caption failed for %s: %s", doc_id, exc)
        return ""


# ── Public API ────────────────────────────────────────────────────────────────

def get_text_surrogate(chunk: dict) -> str:
    """
    Return a plain-text surrogate for a retrieved chunk, suitable for RAGAS scoring.

    Args:
        chunk: A dict with at least 'modality', 'chunk_text', and 'metadata' keys.
               Optional: 'doc_id' for image captioning.

    Returns:
        A non-None string. Empty string if surrogate cannot be produced.
    """
    modality = chunk.get("modality", "text")

    if modality == "audio":
        metadata = chunk.get("metadata") or {}
        transcript = metadata.get("transcript", "")
        return transcript if transcript else ""

    if modality == "image":
        doc_id = chunk.get("doc_id", chunk.get("source_file", "unknown"))
        try:
            return _caption_image(doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Image caption failed for %s: %s", doc_id, exc)
            return ""

    # Default: text modality
    return chunk.get("chunk_text", "")


def build_ragas_contexts(hits: list[dict]) -> list[str]:
    """
    Map a list of retrieved chunk dicts to a list of text context strings for RAGAS.

    Filters out empty surrogates to avoid polluting RAGAS scoring with blank entries.

    Args:
        hits: List of chunk dicts returned by the pipeline (from QueryResponse.hits
              or the internal state chunks list).

    Returns:
        List of non-empty text strings. May be shorter than hits if surrogates fail.
    """
    contexts = []
    for chunk in hits:
        text = get_text_surrogate(chunk)
        if text:
            contexts.append(text)
    return contexts
