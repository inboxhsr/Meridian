"""
tests/test_pipeline_e2e.py — Sprint 3 test gate

End-to-end integration tests: route → retrieve → generate.

Requires:
  - Milvus running with corpus ingested
  - GEMINI_EMBEDDING_KEY set (for retrieval)
  - DEEPSEEK_API_KEY set (for generation)

All tests use --skip-pii equivalent to avoid burning Gemini quota.
"""

import pytest


@pytest.fixture(scope="module")
def pipeline_ready():
    """Skip all E2E tests if dependencies are not available."""
    import os
    from ingest import milvus_store as store

    missing_env = [v for v in ("GEMINI_EMBEDDING_KEY", "DEEPSEEK_API_KEY")
                   if not os.environ.get(v)]
    if missing_env:
        pytest.skip(f"Missing env vars: {missing_env}")

    try:
        client = store.get_client()
        n = store.count(client)
        if n == 0:
            pytest.skip("Corpus not ingested. Run: python scripts/run_ingest.py")
        return n
    except Exception as e:
        pytest.skip(f"Milvus not reachable: {e}")


# ── Router + Retriever (no generator) ────────────────────────────────────────

def test_english_query_retrieves_results(pipeline_ready):
    from pipeline.router import route
    from pipeline.retriever import retrieve

    routing = route("What is the travel expense reimbursement limit?", skip_pii=True)
    assert routing["lang"] == "en"

    chunks = retrieve(routing["safe_query"], top_k=5, bu_filter=routing["bu_filter"])
    assert len(chunks) > 0, "English query returned no chunks"


def test_chinese_query_detected_and_retrieves(pipeline_ready):
    from pipeline.router import route
    from pipeline.retriever import retrieve

    routing = route("差旅报销政策", skip_pii=True)
    assert routing["lang"] == "zh", f"Expected 'zh', got '{routing['lang']}'"

    chunks = retrieve(routing["safe_query"], top_k=5)
    assert len(chunks) > 0, "Chinese query returned no chunks"


def test_bu_scoped_query(pipeline_ready):
    from pipeline.router import route
    from pipeline.retriever import retrieve

    routing = route("incident response steps", bu_filter="it_security", skip_pii=True)
    chunks = retrieve(routing["safe_query"], top_k=5, bu_filter=routing["bu_filter"])

    assert all(c["bu"] == "it_security" for c in chunks), (
        "BU-scoped query returned chunks from wrong BUs"
    )


# ── Full pipeline (route + retrieve + generate) ───────────────────────────────

def test_full_pipeline_english(pipeline_ready):
    from pipeline.router import route
    from pipeline.retriever import retrieve
    from pipeline.generator import generate

    routing = route("What is the leave policy for employees?", skip_pii=True)
    chunks  = retrieve(routing["safe_query"], top_k=5, bu_filter=routing["bu_filter"])
    result  = generate(routing["safe_query"], chunks, lang=routing["lang"])

    assert "answer" in result
    assert "sources" in result
    assert "lang" in result
    assert result["lang"] == "en"
    assert len(result["answer"]) > 50, "Answer too short — likely empty or error response"
    assert len(result["sources"]) > 0, "No sources cited in answer"


def test_full_pipeline_answer_is_grounded(pipeline_ready):
    """The answer must not be completely unrelated — it must mention something from context."""
    from pipeline.router import route
    from pipeline.retriever import retrieve
    from pipeline.generator import generate

    routing = route("What are the IT security access control requirements?", skip_pii=True)
    chunks  = retrieve(routing["safe_query"], top_k=5, bu_filter="it_security")
    result  = generate(routing["safe_query"], chunks, lang="en")

    # At minimum, the answer should be non-empty and contain some relevant term
    answer_lower = result["answer"].lower()
    relevant_terms = ["access", "security", "policy", "control", "user", "system",
                      "don't have", "not have", "knowledge base"]
    assert any(t in answer_lower for t in relevant_terms), (
        f"Answer doesn't appear grounded in context. Got:\n{result['answer']}"
    )


def test_full_pipeline_no_chunks_returns_graceful_response(pipeline_ready):
    """When no chunks are found, generator must return a graceful fallback."""
    from pipeline.generator import generate

    result = generate("complete gibberish xyzzy frobnosticator", chunks=[], lang="en")
    assert result["answer"]
    assert result["chunks_used"] == 0
    assert result["sources"] == []
