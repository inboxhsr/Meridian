"""
tests/test_ingest_chunking.py — Sprint 2 test gate

Unit tests for the chunker. Pure logic — no API calls, no file I/O.
"""

import pytest
from ingest.chunker import chunk_text, make_records, CHARS_PER_CHUNK, CHARS_OVERLAP


# ── chunk_text ────────────────────────────────────────────────────────────────

def test_empty_input_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_returns_one_chunk():
    text = "Hello world. This is a short document."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert "Hello world" in chunks[0]


def test_long_text_produces_multiple_chunks():
    # Generate text well above CHARS_PER_CHUNK
    text = "This is a sentence. " * 200  # ~4000 chars
    chunks = chunk_text(text)
    assert len(chunks) >= 2


def test_chunks_cover_full_text():
    """Every part of the original text must appear in at least one chunk."""
    text = "Sentence number {i}. " * 100
    text = "".join(f"Sentence number {i}. " for i in range(100))
    chunks = chunk_text(text)
    # First token of text must appear in first chunk
    assert text[:20].strip() in chunks[0]
    # Last token of text must appear in last chunk
    assert text[-20:].strip() in chunks[-1]


def test_overlap_exists():
    """Adjacent chunks must share some content (overlap > 0)."""
    text = "Word " * 600  # 3000 chars, well above one chunk
    chunks = chunk_text(text, chars_per_chunk=500, overlap=100)
    if len(chunks) < 2:
        pytest.skip("Too few chunks to test overlap")
    # The end of chunk[0] should overlap with the start of chunk[1]
    tail = chunks[0][-80:]
    head = chunks[1][:80]
    # At least some words should appear in both
    tail_words = set(tail.split())
    head_words = set(head.split())
    assert tail_words & head_words, "Expected overlap between consecutive chunks"


def test_no_empty_chunks():
    """chunk_text must not return any empty or whitespace-only strings."""
    text = "Para one.\n\n\n\nPara two.\n\n\n\nPara three." * 50
    chunks = chunk_text(text)
    for i, c in enumerate(chunks):
        assert c.strip(), f"Chunk {i} is empty or whitespace-only"


def test_chunk_size_respected():
    """Each chunk must not substantially exceed chars_per_chunk."""
    text = "A" * 10_000
    chunks = chunk_text(text, chars_per_chunk=500, overlap=50)
    for i, c in enumerate(chunks):
        # Allow 20% overshoot for sentence-snapping
        assert len(c) <= 600, f"Chunk {i} length {len(c)} exceeds limit+20%"


# ── make_records ──────────────────────────────────────────────────────────────

def test_make_records_structure():
    """make_records returns list of dicts with all required keys."""
    text = "This document covers expense policies. " * 50
    records = make_records(text, "hr_expense_policy.en.pdf", "hr", "en", "pdf")
    assert len(records) >= 1
    for i, r in enumerate(records):
        assert r["source_file"] == "hr_expense_policy.en.pdf"
        assert r["bu"] == "hr"
        assert r["lang"] == "en"
        assert r["modality"] == "pdf"
        assert r["chunk_index"] == i
        assert "text" in r
        assert "embedding" not in r  # embeddings added later by embedder


def test_make_records_empty_text_returns_empty():
    records = make_records("", "file.pdf", "hr", "en", "pdf")
    assert records == []


def test_make_records_chunk_index_sequential():
    text = "Sentence. " * 300
    records = make_records(text, "test.pdf", "product", "zh", "pdf")
    indices = [r["chunk_index"] for r in records]
    assert indices == list(range(len(records)))
