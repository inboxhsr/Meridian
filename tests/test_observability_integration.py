"""
tests/test_observability_integration.py — Sprint 8

Integration tests for the observability layer wired into the live pipeline.

Requirements:
  - Milvus running with Sprint 7 schema (docker compose up milvus -d)
  - GEMINI_API_KEY_A, DEEPSEEK_API_KEY, GEMINI_EMBEDDING_KEY set in .env
  - Collection meridian_corpus populated (python scripts/run_ingest.py --reset)

Run:
  python -m pytest tests/test_observability_integration.py -v

Per AGENTS.md §7 testing rule: the agent proposes these commands;
the user runs them and pastes the output.

What these tests verify:
  1. A live graph.invoke() writes rows to SQLite for every node in the path
  2. All rows for one invocation share the same query_id
  3. estimated_cost is NEVER NULL — not even for local nodes
  4. intent_classifier row has model_used = 'gemini-2.5-flash'
  5. reranker / retriever rows have estimated_cost = 0.0 exactly
  6. Two separate invocations produce two distinct query_id values
  7. DB row count grows monotonically across invocations
"""

from __future__ import annotations
import sqlite3
import uuid

import pytest

from observability.db import init_db


# ── Skip guard ────────────────────────────────────────────────────────────────

def _milvus_available() -> bool:
    try:
        from ingest import milvus_store as store
        client = store.get_client()
        return client.has_collection(store.COLLECTION)
    except Exception:
        return False


def _env_keys_present() -> bool:
    import os
    return bool(
        os.environ.get("GEMINI_API_KEY_A")
        and os.environ.get("DEEPSEEK_API_KEY")
        and os.environ.get("GEMINI_EMBEDDING_KEY")
    )


pytestmark = pytest.mark.skipif(
    not (_milvus_available() and _env_keys_present()),
    reason=(
        "Integration tests require Milvus running + API keys set. "
        "Start Docker, ensure .env is populated, then re-run."
    ),
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def _graph():
    """Compile the LangGraph graph once for all tests in this module."""
    from pipeline.graph import graph
    return graph


@pytest.fixture(scope="module")
def live_db(tmp_path_factory):
    """
    Use a dedicated temp SQLite DB for the integration tests so they don't
    pollute the production observability/meridian.db.
    """
    db_file = tmp_path_factory.mktemp("obs_integration") / "test_live.db"
    import observability.db as db_mod
    original = db_mod._DB_PATH
    db_mod._DB_PATH = db_file
    init_db()
    yield db_file
    db_mod._DB_PATH = original


def _invoke(graph, query: str, bu_filter: str = "", top_k: int = 5,
            session_id: str | None = None) -> tuple[dict, str]:
    """Invoke the graph with a full initial state including query_id.

    Returns (result_dict, session_id) so tests can query the DB by session.
    """
    sid = session_id or str(uuid.uuid4())
    initial_state = {
        "query_id":           sid,
        "query":              query,
        "bu_filter":          bu_filter,
        "top_k":              top_k,
        "lang":               "en",
        "intent":             "",
        "safe_query":         query,
        "pii_flagged":        False,
        "sub_questions":      [],
        "chunks":             [],
        "retrieval_round":    0,
        "groundedness_score": 0.0,
        "relevance_score":    0.0,
        "verdict":            "",
        "critic_reasoning":   "",
        "answer":             "",
        "sources":            [],
        "chunks_used":        0,
        "abstained":          False,
    }
    return graph.invoke(initial_state), sid


# ── Helper ────────────────────────────────────────────────────────────────────

def _rows_for_session(db_path, query_id: str) -> list[dict]:
    """Return all query_log rows for a given query_id as a list of dicts."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM query_log WHERE query_id = ? ORDER BY id",
        (query_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _all_rows(db_path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM query_log ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_invoke_writes_rows_to_sqlite(_graph, live_db):
    """A full graph.invoke() must produce at least one row in query_log."""
    _, sid = _invoke(_graph, "What is the travel expense reimbursement limit?", bu_filter="hr")
    rows = _rows_for_session(live_db, sid)
    assert len(rows) > 0, "Expected at least one observability row after graph.invoke()"


def test_all_rows_share_query_id(_graph, live_db):
    """All rows written during one graph.invoke() must share the same query_id."""
    # Inject a known session_id so we can look up rows precisely.
    # Note: monkey-patching _db_module.log_node does NOT work here because each
    # pipeline node imported log_node by name at module load time — the module
    # attribute patch has no effect on those already-bound references.
    _, sid = _invoke(
        _graph,
        "What is the IT security access control policy?",
        bu_filter="it_security",
    )
    rows = _rows_for_session(live_db, sid)

    assert len(rows) > 0, (
        f"log_node was never called for session {sid} — nodes are not logging"
    )
    unique_ids = {r["query_id"] for r in rows}
    assert unique_ids == {sid}, (
        f"Expected all rows to carry query_id={sid!r}; got: {unique_ids}"
    )


def test_estimated_cost_never_null(_graph, live_db):
    """After a live invoke, every row in the DB must have estimated_cost IS NOT NULL."""
    _, sid = _invoke(_graph, "What is the leave policy entitlement?", bu_filter="hr")
    rows = _rows_for_session(live_db, sid)
    for row in rows:
        assert row["estimated_cost"] is not None, (
            f"NULL estimated_cost found for node={row['node_name']}, id={row['id']}"
        )


def test_intent_classifier_row_has_correct_model(_graph, live_db):
    """The intent_classifier row must record model_used = 'gemini-2.5-flash'."""
    _, sid = _invoke(
        _graph, "Compare the expense policy and IT security incident response", bu_filter=""
    )
    rows = _rows_for_session(live_db, sid)
    classifier_rows = [r for r in rows if r["node_name"] == "intent_classifier"]
    assert len(classifier_rows) >= 1, "No intent_classifier row found after invoke"
    assert classifier_rows[0]["model_used"] == "gemini-2.5-flash", (
        f"Expected gemini-2.5-flash, got: {classifier_rows[0]['model_used']}"
    )


def test_local_nodes_have_zero_cost(_graph, live_db):
    """Reranker and retriever rows must have estimated_cost = 0.0 exactly."""
    _, sid = _invoke(_graph, "What is the expense reimbursement cap?", bu_filter="hr")
    rows = _rows_for_session(live_db, sid)

    zero_cost_nodes = {"reranker", "retriever", "abstainer", "generator", "query_rewriter"}
    for row in rows:
        if row["node_name"] in zero_cost_nodes:
            assert row["estimated_cost"] == 0.0, (
                f"Expected cost=0.0 for {row['node_name']}, got {row['estimated_cost']}"
            )


def test_two_invocations_produce_distinct_query_ids(_graph, live_db):
    """Two separate graph.invoke() calls must log rows under different query_ids."""
    # Inject pre-known session IDs so we can verify isolation via the DB.
    # Monkey-patching log_node does NOT work (nodes imported it by name at load time).
    sid_a = str(uuid.uuid4())
    sid_b = str(uuid.uuid4())

    _invoke(_graph, "What is the travel expense limit?",   bu_filter="hr",          session_id=sid_a)
    _invoke(_graph, "What is the IT security policy?",     bu_filter="it_security", session_id=sid_b)

    rows_a = _rows_for_session(live_db, sid_a)
    rows_b = _rows_for_session(live_db, sid_b)

    assert len(rows_a) > 0, f"No rows found for first invocation (sid={sid_a})"
    assert len(rows_b) > 0, f"No rows found for second invocation (sid={sid_b})"
    assert sid_a != sid_b, "Session IDs must be distinct (uuid4 collision — extremely unlikely)"
    # Confirm no cross-contamination: rows_a contain only sid_a, rows_b only sid_b
    assert all(r["query_id"] == sid_a for r in rows_a), "rows_a contains a foreign query_id"
    assert all(r["query_id"] == sid_b for r in rows_b), "rows_b contains a foreign query_id"


def test_row_count_grows_across_invocations(_graph, live_db):
    """Each graph.invoke() must add rows — DB grows monotonically."""
    count_before = len(_all_rows(live_db))
    _invoke(_graph, "What are the PTO allowances under the leave policy?", bu_filter="hr")
    count_after = len(_all_rows(live_db))
    assert count_after > count_before, (
        f"Row count did not grow: before={count_before}, after={count_after}"
    )
