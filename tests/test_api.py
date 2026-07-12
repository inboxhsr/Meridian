"""
tests/test_api.py — Sprint 4 test gate

FastAPI endpoint tests using TestClient (synchronous, no live server needed).

Requires: Milvus running + corpus ingested + GEMINI_EMBEDDING_KEY + DEEPSEEK_API_KEY.
"""

import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a TestClient. Skip all tests if dependencies are missing."""
    missing = [v for v in ("GEMINI_EMBEDDING_KEY", "DEEPSEEK_API_KEY") if not os.environ.get(v)]
    if missing:
        pytest.skip(f"Missing env vars: {missing}")

    try:
        from ingest import milvus_store as store
        mv = store.get_client()
        if store.count(mv) == 0:
            pytest.skip("Corpus not ingested. Run: python scripts/run_ingest.py")
    except Exception as e:
        pytest.skip(f"Milvus not reachable: {e}")

    from app.main import app
    return TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_response_structure(client):
    data = client.get("/health").json()
    for field in ("status", "milvus", "corpus_chunks", "version"):
        assert field in data, f"Missing field: {field}"


def test_health_milvus_connected(client):
    data = client.get("/health").json()
    assert data["milvus"] is True, "Milvus should be reachable during tests"
    assert data["corpus_chunks"] >= 50, (
        f"Expected >= 50 chunks, got {data['corpus_chunks']}. "
        "Run: python scripts/run_ingest.py"
    )


# ── Root ──────────────────────────────────────────────────────────────────────

def test_root_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Meridian" in resp.json().get("service", "")


# ── Query — validation ────────────────────────────────────────────────────────

def test_query_empty_string_fails_422(client):
    """Empty query must fail Pydantic validation."""
    resp = client.post("/query", json={"query": ""})
    assert resp.status_code == 422


def test_query_missing_body_fails_422(client):
    resp = client.post("/query", json={})
    assert resp.status_code == 422


def test_query_invalid_bu_fails_422(client):
    resp = client.post("/query", json={"query": "test", "bu": "finance"})
    assert resp.status_code == 422


# ── Query — successful responses ──────────────────────────────────────────────

def test_query_english_returns_200(client):
    resp = client.post("/query", json={
        "query": "What is the travel expense reimbursement limit?",
        "skip_pii": True,
    })
    assert resp.status_code == 200


def test_query_response_has_all_fields(client):
    resp = client.post("/query", json={"query": "leave policy", "skip_pii": True})
    assert resp.status_code == 200
    data = resp.json()
    for field in ("query", "safe_query", "lang", "pii_flagged", "answer", "sources", "chunks_used", "hits"):
        assert field in data, f"Missing response field: {field}"


def test_query_answer_is_non_empty(client):
    resp = client.post("/query", json={"query": "IT security access control", "skip_pii": True})
    assert resp.status_code == 200
    assert len(resp.json()["answer"]) > 20


def test_query_chinese_detected(client):
    resp = client.post("/query", json={"query": "差旅报销政策", "skip_pii": True})
    assert resp.status_code == 200
    assert resp.json()["lang"] == "zh"


def test_query_bu_filter_applied(client):
    """All returned hits must be from the requested BU."""
    resp = client.post("/query", json={
        "query": "access control policy",
        "bu": "it_security",
        "skip_pii": True,
    })
    assert resp.status_code == 200
    for hit in resp.json()["hits"]:
        assert hit["bu"] == "it_security", (
            f"Hit from wrong BU: {hit['bu']} ({hit['source_file']})"
        )


def test_query_top_k_respected(client):
    resp = client.post("/query", json={"query": "policy", "top_k": 2, "skip_pii": True})
    assert resp.status_code == 200
    assert len(resp.json()["hits"]) <= 2


def test_query_hits_have_score(client):
    resp = client.post("/query", json={"query": "product roadmap", "skip_pii": True})
    assert resp.status_code == 200
    for hit in resp.json()["hits"]:
        assert isinstance(hit["score"], float)
        assert 0.0 <= hit["score"] <= 1.0
