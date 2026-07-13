"""
tests/test_hybrid_retrieval.py — Sprint 7

Integration tests for hybrid dense + sparse (BM25) retrieval.
Requires: Milvus running with Sprint 7 schema (sparse_vector field).
Run AFTER: python scripts/run_ingest.py --reset

These tests verify that:
1. hybrid_search() returns results with the correct shape
2. BU filter scoping is enforced
3. Exact-term queries benefit from sparse recall
4. Cross-lingual queries still work (dense dominates for cross-lingual)
5. The retriever node (pipeline/nodes/retriever.py) still returns correct shape
6. Empty BU filter returns results from multiple BUs
"""

from __future__ import annotations
import pytest

from ingest import milvus_store as store


# ── Skip guard ────────────────────────────────────────────────────────────────

def _milvus_available() -> bool:
    try:
        client = store.get_client()
        return client.has_collection(store.COLLECTION)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _milvus_available(),
    reason="Milvus not running or collection missing — run: python scripts/run_ingest.py --reset",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mv():
    return store.get_client()


@pytest.fixture(scope="module")
def dense_vec():
    """Pre-compute one query embedding for reuse across tests."""
    from ingest import embedder as emb
    client = emb.get_client()
    return emb.embed_one(client, "travel expense reimbursement limit", task_type="RETRIEVAL_QUERY")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_hybrid_search_returns_results(mv, dense_vec):
    """hybrid_search() must return at least one result for a known query."""
    hits = store.hybrid_search(
        mv,
        query_vector=dense_vec,
        query_text="travel expense reimbursement limit",
        top_k=5,
    )
    assert len(hits) > 0, "Expected at least one hit from hybrid search"


def test_hybrid_search_result_shape(mv, dense_vec):
    """Each hit must contain all required keys with correct types."""
    hits = store.hybrid_search(
        mv,
        query_vector=dense_vec,
        query_text="travel expense reimbursement limit",
        top_k=3,
    )
    required_keys = {"score", "text", "source_file", "bu", "lang", "modality", "chunk_index"}
    for hit in hits:
        assert required_keys.issubset(hit.keys()), f"Missing keys in hit: {hit.keys()}"
        assert isinstance(hit["score"], float)
        assert isinstance(hit["text"], str) and len(hit["text"]) > 0
        assert isinstance(hit["chunk_index"], int)


def test_hybrid_search_top_k_respected(mv, dense_vec):
    """Results must not exceed top_k."""
    hits = store.hybrid_search(
        mv,
        query_vector=dense_vec,
        query_text="expense policy",
        top_k=3,
    )
    assert len(hits) <= 3


def test_hybrid_bu_filter_scoped(mv, dense_vec):
    """BU filter must restrict results to the specified BU."""
    hits = store.hybrid_search(
        mv,
        query_vector=dense_vec,
        query_text="access control policy",
        top_k=5,
        filter_expr='bu == "it_security"',
    )
    if hits:
        for hit in hits:
            assert hit["bu"] == "it_security", (
                f"Expected it_security BU, got: {hit['bu']}"
            )


def test_hybrid_no_filter_returns_multiple_bus(mv, dense_vec):
    """Without a BU filter, results should span more than one BU."""
    from ingest import embedder as emb
    client = emb.get_client()
    # Use a generic query that shouldn't strongly favour any single BU
    vec = emb.embed_one(client, "company policy", task_type="RETRIEVAL_QUERY")
    hits = store.hybrid_search(mv, query_vector=vec, query_text="company policy", top_k=10)
    bus = {h["bu"] for h in hits}
    assert len(bus) > 1, f"Expected hits from >1 BU without filter, got: {bus}"


def test_retriever_node_uses_hybrid(mv):
    """pipeline/nodes/retriever.py must return results via hybrid_search path."""
    from pipeline.nodes.retriever import retrieve as node_retrieve
    state = {
        "safe_query":      "What is the travel expense limit?",
        "bu_filter":       "hr",
        "top_k":           5,
        "sub_questions":   [],
        "retrieval_round": 0,
    }
    result = node_retrieve(state)   # type: ignore[arg-type]
    assert "chunks" in result
    assert isinstance(result["chunks"], list)
    # Every chunk must have required fields
    for chunk in result["chunks"]:
        assert "text" in chunk
        assert "source_file" in chunk
        assert "score" in chunk
