"""
ingest/embedder.py — Sprint 2

Generates dense vector embeddings using Gemini text-embedding-004 (768-dim).
Uses GEMINI_EMBEDDING_KEY (separate from the generation key).

Embedding model: text-embedding-004
Dimensions:      768
Task type:       RETRIEVAL_DOCUMENT (for corpus indexing)
                 RETRIEVAL_QUERY    (for query-time lookup)
"""

import os
import time
from google import genai
from google.genai import types

EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIM = 768
_DEFAULT_DELAY = 0.05  # seconds between calls — Gemini embedding is 2000 RPM


def get_client() -> genai.Client:
    """Return a Gemini client configured with the embedding key.

    Forces api_version='v1' (stable) — text-embedding-004 is not available
    on the default v1beta endpoint used by the SDK.
    """
    key = os.environ.get("GEMINI_EMBEDDING_KEY")
    if not key:
        raise EnvironmentError("GEMINI_EMBEDDING_KEY not set in .env")
    return genai.Client(api_key=key, http_options={"api_version": "v1"})


def embed_one(client: genai.Client, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Embed a single text string. Returns 768-dim float list."""
    # Gemini embedding max input is ~8k tokens; truncate at 8000 chars as safe proxy
    text = text[:8000]
    resp = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return list(resp.embeddings[0].values)


def embed_records(
    client: genai.Client,
    records: list[dict],
    delay: float = _DEFAULT_DELAY,
    on_progress=None,
) -> list[dict]:
    """Add 'embedding' field to each record dict in-place. Returns the same list.

    Args:
        records:     List of chunk dicts (must have 'text' key)
        delay:       Seconds to sleep between API calls
        on_progress: Optional callable(done, total) for progress reporting
    """
    total = len(records)
    for i, rec in enumerate(records):
        rec["embedding"] = embed_one(client, rec["text"])
        if delay:
            time.sleep(delay)
        if on_progress:
            on_progress(i + 1, total)
    return records
