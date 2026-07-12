"""
ingest/chunker.py — Sprint 2

Splits extracted text into overlapping chunks suitable for embedding.

Strategy:
  - Character-based sizing (proxy for tokens: 1 token ≈ 4 chars)
  - Tries to break at sentence boundaries (. ? ! 。) to preserve semantics
  - Yields dicts with 'text', 'chunk_index', and pass-through metadata
"""

from __future__ import annotations
import re

# ~300 words / ~400 tokens at 4 chars/token
CHARS_PER_CHUNK: int = 1_200
# ~50-word overlap
CHARS_OVERLAP: int = 200


def chunk_text(
    text: str,
    chars_per_chunk: int = CHARS_PER_CHUNK,
    overlap: int = CHARS_OVERLAP,
) -> list[str]:
    """Split text into overlapping string chunks. Returns list of chunk strings."""
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    stride = chars_per_chunk - overlap  # how far to advance each step

    while start < len(text):
        end = min(start + chars_per_chunk, len(text))
        chunk = text[start:end]

        # Try to snap the end to the nearest sentence boundary
        if end < len(text):
            # Look for sentence-ending punctuation followed by whitespace
            match = None
            for pattern in (r"[.!?。！？]\s", r"[.!?。！？]$"):
                # Search within the last 30% of the chunk
                search_start = max(0, len(chunk) - len(chunk) // 3)
                m = None
                for m in re.finditer(pattern, chunk[search_start:]):
                    pass  # keep the last match
                if m:
                    snap = search_start + m.end()
                    if snap > len(chunk) // 2:  # only snap if meaningful
                        chunk = chunk[:snap]
                        break

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        # Advance by stride, not by chunk length, to get overlap
        start += max(stride, len(chunk) - overlap)

    return chunks


def make_records(
    text: str,
    source_file: str,
    bu: str,
    lang: str,
    modality: str,
    chars_per_chunk: int = CHARS_PER_CHUNK,
    overlap: int = CHARS_OVERLAP,
) -> list[dict]:
    """Chunk text and attach metadata. Returns list of record dicts (no embedding yet)."""
    chunks = chunk_text(text, chars_per_chunk, overlap)
    return [
        {
            "source_file": source_file,
            "bu": bu,
            "lang": lang,
            "modality": modality,
            "chunk_index": i,
            "text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]
