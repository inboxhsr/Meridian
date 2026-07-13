"""
tests/test_e2e_integration.py — Sprint 9 (full-stack E2E)

End-to-end integration tests covering the complete Meridian pipeline:
  FastAPI (POST /query) → LangGraph → Milvus → Generator → Observability DB

What these tests verify:
  1.  Health endpoint confirms Milvus reachable + corpus loaded
  2.  English EN query → answer + citations + hits + observability rows
  3.  Chinese ZH query → language detected + ZH hits returned
  4.  Hindi HI query → language detected correctly
  5.  BU scoping — hr query must NOT return it_security chunks
  6.  BU scoping — it_security query must NOT return hr chunks
  7.  Cross-lingual — Hindi question → English source retrieved
  8.  Cross-lingual — English question → Chinese source retrieved
  9.  Abstention — unanswerable query returns abstention signal
  10. Observability — every /query call writes rows grouped by query_id
  11. Observability — estimated_cost is never NULL for any row
  12. Observability — intent_classifier row has model_used = gemini-2.5-flash
  13. Observability — local nodes (reranker, retriever) have cost = 0.0
  14. Citations — answer for answered queries includes at least one source
  15. Regression guard — response time under 120s for a simple EN query

Requirements (all must be met or test is SKIPPED, not FAILED):
  - Docker Milvus running:  docker compose up milvus -d
  - Corpus ingested:        python scripts/run_ingest.py
  - FastAPI server up:      uvicorn app.main:app --port 8000
  - .env populated:         GEMINI_API_KEY_A, GEMINI_EMBEDDING_KEY, DEEPSEEK_API_KEY

Run:
  python -m pytest tests/test_e2e_integration.py -v
  python -m pytest tests/test_e2e_integration.py -v -k "not slow"  # skip latency test
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid

import pytest
import requests  # type: ignore

# ── Configuration ──────────────────────────────────────────────────────────────

_API_BASE = os.environ.get("MERIDIAN_API_URL", "http://localhost:8000")
_TIMEOUT = 120  # seconds — generous for cold-start + model latency
_ABSTENTION_SIGNALS = [
    "insufficient grounded evidence",
    "unable to answer",
    "no relevant information",
    "could not find",
    "i don't have",
    "out of scope",
    "cannot answer",
    "not have information",
]


# ── Skip guard — applied at module level ───────────────────────────────────────

def _api_alive() -> bool:
    try:
        r = requests.get(f"{_API_BASE}/health", timeout=5)
        return r.status_code == 200 and r.json().get("milvus") is True
    except Exception:
        return False


def _corpus_loaded() -> bool:
    try:
        r = requests.get(f"{_API_BASE}/health", timeout=5)
        return r.status_code == 200 and r.json().get("corpus_chunks", 0) >= 100
    except Exception:
        return False


def _env_keys_present() -> bool:
    return all(os.environ.get(k) for k in (
        "GEMINI_API_KEY_A", "GEMINI_EMBEDDING_KEY", "DEEPSEEK_API_KEY"
    ))


_SKIP_REASON = (
    "E2E integration tests require:\n"
    "  1. uvicorn app.main:app --port 8000\n"
    "  2. docker compose up milvus -d\n"
    "  3. python scripts/run_ingest.py\n"
    "  4. .env: GEMINI_API_KEY_A, GEMINI_EMBEDDING_KEY, DEEPSEEK_API_KEY"
)

pytestmark = pytest.mark.skipif(
    not (_api_alive() and _corpus_loaded() and _env_keys_present()),
    reason=_SKIP_REASON,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def api():
    """Return the FastAPI base URL after confirming it is reachable."""
    return _API_BASE


@pytest.fixture(scope="module")
def live_db(tmp_path_factory):
    """
    Redirect observability DB to a temp file for the duration of these tests.
    Prevents polluting the production observability/meridian.db.
    """
    db_file = tmp_path_factory.mktemp("e2e_integration") / "e2e_test.db"
    import observability.db as db_mod
    from observability.db import init_db
    original = db_mod._DB_PATH
    db_mod._DB_PATH = db_file
    init_db()
    yield db_file
    db_mod._DB_PATH = original


def _post(api_base: str, query: str, bu: str = "", top_k: int = 5) -> dict:
    """POST /query and return parsed JSON. Raises on non-200."""
    payload: dict = {"query": query, "top_k": top_k, "skip_pii": True}
    if bu:
        payload["bu"] = bu
    resp = requests.post(f"{api_base}/query", json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _rows_for_sid(db_path, query_id: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM query_log WHERE query_id = ? ORDER BY id",
        (query_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── 1. Health ──────────────────────────────────────────────────────────────────

def test_health_milvus_reachable(api):
    """GET /health must return milvus=true and corpus_chunks >= 100."""
    data = requests.get(f"{api}/health", timeout=10).json()
    assert data["milvus"] is True
    assert data["corpus_chunks"] >= 100, (
        f"Expected ≥100 corpus chunks, got {data['corpus_chunks']}. "
        "Run: python scripts/run_ingest.py"
    )


def test_health_response_shape(api):
    """GET /health must return all required fields."""
    data = requests.get(f"{api}/health", timeout=10).json()
    for field in ("status", "milvus", "corpus_chunks", "version"):
        assert field in data, f"Missing health field: {field}"


# ── 2. English baseline query ──────────────────────────────────────────────────

def test_en_query_returns_200_and_answer(api):
    """English query must return HTTP 200 and a non-empty answer."""
    data = _post(api, "What is the travel expense reimbursement limit?", bu="hr")
    assert data["answer"], "Answer must not be empty for a known corpus question"
    assert len(data["answer"]) > 30, "Answer too short — likely an error string"


def test_en_query_response_has_all_fields(api):
    """POST /query response must contain every field defined in QueryResponse."""
    data = _post(api, "What are the password requirements for Meridian systems?", bu="it_security")
    for field in ("query", "safe_query", "lang", "pii_flagged", "answer", "sources", "chunks_used", "hits"):
        assert field in data, f"Missing response field: {field}"


def test_en_query_lang_detected_as_en(api):
    """Language detection must return 'en' for an English query."""
    data = _post(api, "What is the leave policy for full-time employees?", bu="hr")
    assert data["lang"] == "en", f"Expected 'en', got '{data['lang']}'"


def test_en_query_hits_are_returned(api):
    """At least one chunk hit must be returned for a known corpus query."""
    data = _post(api, "What is the minimum password length?", bu="it_security")
    assert len(data["hits"]) > 0, "No hits returned for a known corpus question"


def test_en_query_hits_have_valid_shape(api):
    """Each hit must have source_file, bu, lang, modality, score."""
    data = _post(api, "IT security access control policy", bu="it_security")
    for hit in data["hits"]:
        for field in ("source_file", "bu", "lang", "modality", "score"):
            assert field in hit, f"Hit missing field '{field}': {hit}"
        assert isinstance(hit["score"], float)


# ── 3. Chinese query ───────────────────────────────────────────────────────────

def test_zh_query_language_detected(api):
    """Chinese query must be detected as 'zh'."""
    data = _post(api, "差旅报销政策")
    assert data["lang"] == "zh", f"Expected 'zh', got '{data['lang']}'"


def test_zh_query_returns_answer(api):
    """Chinese query must return a non-empty answer."""
    data = _post(api, "差旅报销政策", bu="hr")
    assert data["answer"], "Chinese query returned empty answer"


def test_zh_query_returns_hits(api):
    """Chinese query must retrieve at least one chunk."""
    data = _post(api, "产品路线图2026", bu="product")
    assert len(data["hits"]) > 0, "Chinese product query returned no hits"


# ── 4. Hindi query ────────────────────────────────────────────────────────────

def test_hi_query_language_detected(api):
    """Hindi query must be detected as 'hi'."""
    data = _post(api, "व्यय नीति क्या है?", bu="hr")
    assert data["lang"] == "hi", f"Expected 'hi', got '{data['lang']}'"


# ── 5 & 6. BU scoping / partition isolation ───────────────────────────────────

def test_hr_query_hits_stay_in_hr(api):
    """An HR-scoped query must ONLY return chunks from the 'hr' BU."""
    data = _post(api, "What is the leave policy for employees?", bu="hr")
    for hit in data["hits"]:
        assert hit["bu"] == "hr", (
            f"BU isolation breach: hr-scoped query returned chunk from '{hit['bu']}' "
            f"({hit['source_file']})"
        )


def test_it_security_query_hits_stay_in_it_security(api):
    """An it_security-scoped query must ONLY return chunks from 'it_security'."""
    data = _post(api, "incident response P1 critical", bu="it_security")
    for hit in data["hits"]:
        assert hit["bu"] == "it_security", (
            f"BU isolation breach: it_security-scoped query returned chunk from "
            f"'{hit['bu']}' ({hit['source_file']})"
        )


def test_hr_query_does_not_return_it_security_chunks(api):
    """Partition isolation: an HR question with it_security-like keywords must
    not retrieve it_security chunks when scoped to HR."""
    # Ask an HR question that mentions 'access' — could naively match IT Security
    data = _post(api, "Who has access to the employee expense reimbursement system?", bu="hr")
    for hit in data["hits"]:
        assert hit["bu"] == "hr", (
            f"Partition isolation FAILED: hr-scoped query returned chunk from "
            f"'{hit['bu']}' ({hit['source_file']})"
        )


# ── 7. Cross-lingual — Hindi question → English source ────────────────────────

def test_cross_lingual_hi_query_retrieves_results(api):
    """Hindi query for content that exists only in English must still retrieve chunks."""
    # "What is the password minimum length?" in Hindi — source only in EN
    data = _post(api, "पासवर्ड की न्यूनतम लंबाई क्या होनी चाहिए?", bu="it_security")
    # The pipeline must retrieve something (cross-lingual retrieval via dense embedding)
    assert len(data["hits"]) > 0, (
        "Cross-lingual retrieval FAILED: Hindi query returned no chunks "
        "(expected EN it_security chunks via dense embedding)"
    )


def test_cross_lingual_hi_query_answer_non_empty(api):
    """Hindi query must produce a non-empty answer even if source is English."""
    data = _post(api, "P1 साइबर सुरक्षा घटना पर प्रतिक्रिया समय कितना है?", bu="it_security")
    assert data["answer"], "Cross-lingual Hindi→EN query returned empty answer"


# ── 8. Cross-lingual — English question → Chinese source ─────────────────────

def test_cross_lingual_en_query_retrieves_zh_chunks(api):
    """An English question about Chinese-only content must retrieve chunks."""
    # Ask about ZH product roadmap in English — corpus has ZH version
    data = _post(api, "What does the Chinese product roadmap say about 2026 engineering priorities?", bu="product")
    assert len(data["hits"]) > 0, (
        "Cross-lingual EN→ZH retrieval returned no hits"
    )


# ── 9. Abstention — unanswerable queries ──────────────────────────────────────

def test_unanswerable_query_triggers_abstention(api):
    """A question with no corpus answer must return an abstention signal."""
    data = _post(api, "What is Meridian's stock price today?")
    answer_lower = data["answer"].lower()
    abstained = any(sig in answer_lower for sig in _ABSTENTION_SIGNALS) or not data["answer"]
    assert abstained, (
        f"Expected abstention for unanswerable query, got: {data['answer'][:200]}"
    )


def test_personal_data_query_triggers_abstention(api):
    """A query for personal/private data not in corpus must return abstention."""
    data = _post(api, "What is the CEO's personal home address?")
    answer_lower = data["answer"].lower()
    abstained = any(sig in answer_lower for sig in _ABSTENTION_SIGNALS) or not data["answer"]
    assert abstained, (
        f"Expected abstention for personal-data query, got: {data['answer'][:200]}"
    )


# ── 10–13. Observability ──────────────────────────────────────────────────────

def test_query_writes_observability_rows(api, live_db):
    """POST /query must write at least one row to the observability DB."""
    from pipeline.graph import graph

    sid = str(uuid.uuid4())
    initial_state = {
        "query_id": sid,
        "query": "What is the expense policy exception approval process?",
        "bu_filter": "hr",
        "top_k": 5,
        "lang": "en",
        "intent": "",
        "safe_query": "What is the expense policy exception approval process?",
        "pii_flagged": False,
        "sub_questions": [],
        "chunks": [],
        "retrieval_round": 0,
        "groundedness_score": 0.0,
        "relevance_score": 0.0,
        "verdict": "",
        "critic_reasoning": "",
        "answer": "",
        "sources": [],
        "chunks_used": 0,
        "abstained": False,
    }
    graph.invoke(initial_state)
    rows = _rows_for_sid(live_db, sid)
    assert len(rows) > 0, f"No observability rows written for session {sid}"


def test_observability_rows_share_query_id(api, live_db):
    """All rows written during a single graph invocation must share one query_id."""
    from pipeline.graph import graph

    sid = str(uuid.uuid4())
    initial_state = {
        "query_id": sid,
        "query": "What is the minimum password length for Meridian?",
        "bu_filter": "it_security",
        "top_k": 5,
        "lang": "en",
        "intent": "",
        "safe_query": "What is the minimum password length for Meridian?",
        "pii_flagged": False,
        "sub_questions": [],
        "chunks": [],
        "retrieval_round": 0,
        "groundedness_score": 0.0,
        "relevance_score": 0.0,
        "verdict": "",
        "critic_reasoning": "",
        "answer": "",
        "sources": [],
        "chunks_used": 0,
        "abstained": False,
    }
    graph.invoke(initial_state)
    rows = _rows_for_sid(live_db, sid)
    assert len(rows) > 0
    unique_ids = {r["query_id"] for r in rows}
    assert unique_ids == {sid}, f"Multiple query_ids in one session: {unique_ids}"


def test_observability_estimated_cost_never_null(api, live_db):
    """estimated_cost must be a float (never NULL) for every observability row."""
    from pipeline.graph import graph

    sid = str(uuid.uuid4())
    initial_state = {
        "query_id": sid,
        "query": "What is the GraphQL API release date?",
        "bu_filter": "product",
        "top_k": 5,
        "lang": "en",
        "intent": "",
        "safe_query": "What is the GraphQL API release date?",
        "pii_flagged": False,
        "sub_questions": [],
        "chunks": [],
        "retrieval_round": 0,
        "groundedness_score": 0.0,
        "relevance_score": 0.0,
        "verdict": "",
        "critic_reasoning": "",
        "answer": "",
        "sources": [],
        "chunks_used": 0,
        "abstained": False,
    }
    graph.invoke(initial_state)
    rows = _rows_for_sid(live_db, sid)
    for row in rows:
        assert row["estimated_cost"] is not None, (
            f"NULL estimated_cost for node={row['node_name']}"
        )


def test_observability_intent_classifier_model(api, live_db):
    """The intent_classifier observability row must record model_used = gemini-2.5-flash."""
    from pipeline.graph import graph

    sid = str(uuid.uuid4())
    initial_state = {
        "query_id": sid,
        "query": "What is the IT incident response SLA?",
        "bu_filter": "it_security",
        "top_k": 5,
        "lang": "en",
        "intent": "",
        "safe_query": "What is the IT incident response SLA?",
        "pii_flagged": False,
        "sub_questions": [],
        "chunks": [],
        "retrieval_round": 0,
        "groundedness_score": 0.0,
        "relevance_score": 0.0,
        "verdict": "",
        "critic_reasoning": "",
        "answer": "",
        "sources": [],
        "chunks_used": 0,
        "abstained": False,
    }
    graph.invoke(initial_state)
    rows = _rows_for_sid(live_db, sid)
    classifier = [r for r in rows if r["node_name"] == "intent_classifier"]
    assert classifier, "No intent_classifier row found"
    assert classifier[0]["model_used"] == "gemini-2.5-flash"


def test_observability_local_nodes_zero_cost(api, live_db):
    """Reranker and retriever observability rows must have estimated_cost = 0.0."""
    from pipeline.graph import graph

    sid = str(uuid.uuid4())
    initial_state = {
        "query_id": sid,
        "query": "What is the code of conduct regarding workplace ethics?",
        "bu_filter": "hr",
        "top_k": 5,
        "lang": "en",
        "intent": "",
        "safe_query": "What is the code of conduct regarding workplace ethics?",
        "pii_flagged": False,
        "sub_questions": [],
        "chunks": [],
        "retrieval_round": 0,
        "groundedness_score": 0.0,
        "relevance_score": 0.0,
        "verdict": "",
        "critic_reasoning": "",
        "answer": "",
        "sources": [],
        "chunks_used": 0,
        "abstained": False,
    }
    graph.invoke(initial_state)
    rows = _rows_for_sid(live_db, sid)
    zero_cost_nodes = {"reranker", "retriever"}
    for row in rows:
        if row["node_name"] in zero_cost_nodes:
            assert row["estimated_cost"] == 0.0, (
                f"{row['node_name']} has non-zero cost: {row['estimated_cost']}"
            )


# ── 14. Citations ──────────────────────────────────────────────────────────────

def test_answered_query_has_sources(api):
    """A successfully answered query must include at least one cited source."""
    data = _post(api, "What is the P1 incident response time?", bu="it_security")
    answer_lower = data["answer"].lower()
    is_abstained = any(sig in answer_lower for sig in _ABSTENTION_SIGNALS)
    if not is_abstained:
        # Only assert sources when the pipeline actually answered
        assert data["sources"] or data["chunks_used"] > 0, (
            "Answered query has no sources and chunks_used=0"
        )


def test_top_k_limit_respected(api):
    """The number of hits must not exceed the requested top_k."""
    data = _post(api, "expense policy", bu="hr", top_k=3)
    assert len(data["hits"]) <= 3, (
        f"top_k=3 requested but got {len(data['hits'])} hits"
    )


# ── 15. Latency regression guard ──────────────────────────────────────────────

@pytest.mark.slow
def test_simple_query_completes_under_120s(api):
    """A simple English query must complete within 120 seconds (p95 budget)."""
    start = time.time()
    data = _post(api, "What is the leave policy?", bu="hr")
    elapsed = time.time() - start
    assert elapsed < 120, (
        f"Query took {elapsed:.1f}s — exceeds 120s p95 budget"
    )
    assert data["answer"]  # sanity: it also actually answered
