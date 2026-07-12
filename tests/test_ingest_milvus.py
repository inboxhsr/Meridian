"""
tests/test_ingest_milvus.py — Sprint 2 test gate

Integration tests that verify the corpus has been successfully ingested into Milvus.

PREREQUISITE: Run the ingest pipeline first:
    python scripts/run_ingest.py

These tests connect to the live Milvus instance. Milvus must be running.
"""

import os
import pytest
from pathlib import Path

# Minimum chunks expected after a successful ingest (PDFs only, no OCR).
# 25 PDFs × ~3 chunks each = ~75; using 50 as a conservative lower bound.
MIN_CHUNKS = 50

# All 4 business units must be represented
EXPECTED_BUS = {"hr", "it_security", "product", "exec_comms"}

# All 3 languages must appear in the PDF chunks
EXPECTED_LANGS = {"en", "hi", "zh"}


@pytest.fixture(scope="module")
def mv():
    """Return a connected MilvusClient. Skip all tests if Milvus unreachable."""
    from ingest.milvus_store import get_client, COLLECTION
    try:
        client = get_client()
        # Probe connection
        client.has_collection(COLLECTION)
        return client
    except Exception as e:
        pytest.skip(f"Milvus not reachable: {e}")


# ── Collection existence ──────────────────────────────────────────────────────

def test_collection_exists(mv):
    """meridian_corpus collection must exist after ingest."""
    from ingest.milvus_store import COLLECTION
    assert mv.has_collection(COLLECTION), (
        f"Collection '{COLLECTION}' not found.\n"
        "Run: python scripts/run_ingest.py"
    )


# ── Record count ──────────────────────────────────────────────────────────────

def test_minimum_chunk_count(mv):
    """Collection must have at least MIN_CHUNKS records."""
    from ingest.milvus_store import count
    n = count(mv)
    assert n >= MIN_CHUNKS, (
        f"Collection has {n} records, need at least {MIN_CHUNKS}.\n"
        "Run: python scripts/run_ingest.py"
    )


# ── Business unit coverage ────────────────────────────────────────────────────

def test_all_business_units_ingested(mv):
    """All 4 business units must have at least one chunk in Milvus."""
    from ingest.milvus_store import COLLECTION
    missing = []
    for bu in EXPECTED_BUS:
        res = mv.query(
            COLLECTION,
            filter=f'bu == "{bu}"',
            output_fields=["id"],
            limit=1,
        )
        if not res:
            missing.append(bu)
    assert not missing, (
        f"Missing business units in Milvus: {sorted(missing)}.\n"
        "Run: python scripts/run_ingest.py"
    )


# ── Language coverage ─────────────────────────────────────────────────────────

def test_all_languages_ingested(mv):
    """EN, HI, ZH must all have chunks in Milvus."""
    from ingest.milvus_store import COLLECTION
    missing = []
    for lang in EXPECTED_LANGS:
        res = mv.query(
            COLLECTION,
            filter=f'lang == "{lang}"',
            output_fields=["id"],
            limit=1,
        )
        if not res:
            missing.append(lang)
    assert not missing, (
        f"Missing languages in Milvus: {sorted(missing)}.\n"
        "Run: python scripts/run_ingest.py"
    )


# ── Embedding sanity ──────────────────────────────────────────────────────────

def test_embeddings_have_correct_dimension(mv):
    """Sample 5 records and verify each embedding is 3072-dimensional (gemini-embedding-001)."""
    from ingest.milvus_store import COLLECTION, DIM
    res = mv.query(COLLECTION, filter="", output_fields=["embedding"], limit=5)
    assert res, "No records returned from Milvus"
    for row in res:
        vec = row.get("embedding")
        assert vec is not None, "embedding field is None"
        assert len(vec) == DIM, (
            f"Expected embedding dim {DIM}, got {len(vec)}"
        )


# ── Semantic search smoke test ────────────────────────────────────────────────

def test_search_returns_results(mv):
    """A nearest-neighbour search must return at least 1 result."""
    from ingest.milvus_store import COLLECTION, DIM
    # Use a zero vector — not semantically meaningful, but proves the search path works
    dummy_vector = [0.0] * DIM  # 3072 dims for gemini-embedding-001
    results = mv.search(
        collection_name=COLLECTION,
        data=[dummy_vector],
        limit=3,
        output_fields=["text", "source_file"],
    )
    assert results and len(results[0]) > 0, (
        "Milvus search returned no results. "
        "The collection may be empty or the index not built."
    )
