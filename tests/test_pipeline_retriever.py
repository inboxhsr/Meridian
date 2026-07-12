"""
tests/test_pipeline_retriever.py — Sprint 3 test gate

Integration tests for the retriever.
Requires: Milvus running + corpus ingested (run scripts/run_ingest.py first).
Requires: GEMINI_EMBEDDING_KEY set in .env.
"""

import pytest


@pytest.fixture(scope="module")
def milvus_ready():
    """Skip all tests if Milvus is unreachable or collection is empty."""
    from ingest import milvus_store as store
    try:
        client = store.get_client()
        n = store.count(client)
        if n == 0:
            pytest.skip(
                "meridian_corpus collection is empty. "
                "Run: python scripts/run_ingest.py"
            )
        return n
    except Exception as e:
        pytest.skip(f"Milvus not reachable: {e}")


# ── Structure tests ───────────────────────────────────────────────────────────

def test_retrieve_returns_list(milvus_ready):
    from pipeline.retriever import retrieve
    results = retrieve("expense reimbursement policy", top_k=3, skip_pii=True if False else False)
    # retrieve() doesn't have skip_pii; just call it directly
    results = retrieve("expense reimbursement policy", top_k=3)
    assert isinstance(results, list)


def test_retrieve_result_has_required_keys(milvus_ready):
    from pipeline.retriever import retrieve
    results = retrieve("security incident response", top_k=3)
    assert len(results) > 0, "Expected at least 1 result"
    required = {"score", "text", "source_file", "bu", "lang", "modality", "chunk_index"}
    for hit in results:
        missing = required - set(hit.keys())
        assert not missing, f"Hit missing keys: {missing}"


def test_retrieve_text_non_empty(milvus_ready):
    from pipeline.retriever import retrieve
    results = retrieve("annual review presentation", top_k=3)
    for hit in results:
        assert hit["text"].strip(), f"Empty text in hit from {hit['source_file']}"


def test_retrieve_top_k_respected(milvus_ready):
    from pipeline.retriever import retrieve
    for k in (1, 3, 5):
        results = retrieve("product roadmap features", top_k=k)
        assert len(results) <= k, f"Expected <= {k} results, got {len(results)}"


def test_retrieve_scores_are_floats(milvus_ready):
    from pipeline.retriever import retrieve
    results = retrieve("leave policy rules", top_k=3)
    for hit in results:
        assert isinstance(hit["score"], float), f"score is not float: {type(hit['score'])}"


# ── BU filter tests ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("bu", ["hr", "it_security", "product"])
def test_retrieve_bu_filter_respected(bu, milvus_ready):
    """All returned chunks must belong to the requested BU."""
    from pipeline.retriever import retrieve
    results = retrieve("policy and procedures", top_k=5, bu_filter=bu)
    for hit in results:
        assert hit["bu"] == bu, (
            f"BU filter '{bu}' violated: got chunk from bu='{hit['bu']}' "
            f"in file {hit['source_file']}"
        )


def test_retrieve_no_filter_returns_multiple_bus(milvus_ready):
    """Without a BU filter, results should span more than one BU."""
    from pipeline.retriever import retrieve
    results = retrieve("company policies and guidelines", top_k=10)
    bus = {hit["bu"] for hit in results}
    assert len(bus) >= 2, (
        f"Expected results from multiple BUs without filter, got: {bus}"
    )
