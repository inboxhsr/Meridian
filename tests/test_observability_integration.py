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

import observability.db as _db_module
from observability.db import init_db, log_node


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


def _invoke(graph, query: str, bu_filter: str = "", top_k: int = 5) -> dict:
    """Invoke the graph with a full initial state including query_id."""
    initial_state = {
        "query_id":           str(uuid.uuid4()),
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
    return graph.invoke(initial_state)


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
    result = _invoke(_graph, "What is the travel expense reimbursement limit?", bu_filter="hr")
    sid = result.get("query_id") or _all_rows(live_db)[-1]["query_id"]
    rows = _rows_for_session(live_db, sid) if result.get("query_id") else _all_rows(live_db)
    assert len(rows) > 0, "Expected at least one observability row after graph.invoke()"


def test_all_rows_share_query_id(_graph, live_db):
    """All rows written during one graph.invoke() must share the same query_id."""
    session_id = str(uuid.uuid4())

    # Patch log_node to capture the query_id used
    captured_ids: list[str] = []
    original_log = _db_module.log_node

    def _capturing_log_node(*, query_id, **kwargs):
        captured_ids.append(query_id)
        return original_log(query_id=query_id, **kwargs)

    _db_module.log_node = _capturing_log_node
    try:
        _invoke(_graph, "What is the IT security access control policy?", bu_filter="it_security")
    finally:
        _db_module.log_node = original_log

    assert len(captured_ids) > 0, "log_node was never called — nodes not logging"
    unique_ids = set(captured_ids)
    assert len(unique_ids) == 1, (
        f"Expected one unique query_id per invocation; got: {unique_ids}"
    )


def test_estimated_cost_never_null(_graph, live_db):
    """After a live invoke, every row in the DB must have estimated_cost IS NOT NULL."""
    _invoke(_graph, "What is the leave policy entitlement?", bu_filter="hr")
    rows = _all_rows(live_db)
    for row in rows:
        assert row["estimated_cost"] is not None, (
            f"NULL estimated_cost found for node={row['node_name']}, id={row['id']}"
        )


def test_intent_classifier_row_has_correct_model(_graph, live_db):
    """The intent_classifier row must record model_used = 'gemini-2.5-flash'."""
    before_count = len(_all_rows(live_db))
    _invoke(_graph, "Compare the expense policy and IT security incident response", bu_filter="")
    all_rows = _all_rows(live_db)
    new_rows = all_rows[before_count:]

    classifier_rows = [r for r in new_rows if r["node_name"] == "intent_classifier"]
    assert len(classifier_rows) >= 1, "No intent_classifier row found after invoke"
    assert classifier_rows[0]["model_used"] == "gemini-2.5-flash", (
        f"Expected gemini-2.5-flash, got: {classifier_rows[0]['model_used']}"
    )


def test_local_nodes_have_zero_cost(_graph, live_db):
    """Reranker and retriever rows must have estimated_cost = 0.0 exactly."""
    before_count = len(_all_rows(live_db))
    _invoke(_graph, "What is the expense reimbursement cap?", bu_filter="hr")
    all_rows = _all_rows(live_db)
    new_rows = all_rows[before_count:]

    zero_cost_nodes = {"reranker", "retriever", "abstainer", "generator", "query_rewriter"}
    for row in new_rows:
        if row["node_name"] in zero_cost_nodes:
            assert row["estimated_cost"] == 0.0, (
                f"Expected cost=0.0 for {row['node_name']}, got {row['estimated_cost']}"
            )


def test_two_invocations_produce_distinct_query_ids(_graph, live_db):
    """Two separate graph.invoke() calls must log rows under different query_ids."""
    captured: list[list[str]] = [[], []]
    original_log = _db_module.log_node

    call_count = [0]

    def _tracking_log(*, query_id, **kwargs):
        captured[min(call_count[0], 1)].append(query_id)
        return original_log(query_id=query_id, **kwargs)

    # First invocation
    _db_module.log_node = _tracking_log
    try:
        _invoke(_graph, "What is the travel expense limit?", bu_filter="hr")
        call_count[0] = 1
        _invoke(_graph, "What is the IT security policy?", bu_filter="it_security")
    finally:
        _db_module.log_node = original_log

    ids_first  = set(captured[0])
    ids_second = set(captured[1])

    assert len(ids_first) == 1,  "First invocation should use exactly one query_id"
    assert len(ids_second) == 1, "Second invocation should use exactly one query_id"
    assert ids_first != ids_second, (
        f"Two invocations must produce distinct query_ids; both got: {ids_first}"
    )


def test_row_count_grows_across_invocations(_graph, live_db):
    """Each graph.invoke() must add rows — DB grows monotonically."""
    count_before = len(_all_rows(live_db))
    _invoke(_graph, "What are the PTO allowances under the leave policy?", bu_filter="hr")
    count_after = len(_all_rows(live_db))
    assert count_after > count_before, (
        f"Row count did not grow: before={count_before}, after={count_after}"
    )
