"""
tests/test_observability.py — Sprint 8

Unit tests for the observability layer.
All tests use a tmp-path SQLite DB (monkeypatched) — no live APIs, no Milvus.

Test gate: 7/7 must pass to close Sprint 8.
"""

from __future__ import annotations

import sqlite3

import pytest

import observability.db as _db_module


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Redirect all DB operations to a temp file per test for full isolation."""
    db_file = tmp_path / "test_meridian.db"
    monkeypatch.setattr(_db_module, "_DB_PATH", db_file)
    yield db_file


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_sqlite_schema_columns(tmp_db):
    """init_db() must create query_log with all 17 required columns."""
    from observability.db import init_db
    init_db()

    conn = sqlite3.connect(str(tmp_db))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(query_log)")}
    conn.close()

    required = {
        "id", "query_id", "node_name", "retrieval_round", "model_used",
        "groundedness_score", "relevance_score", "verdict", "critic_reasoning",
        "tokens_used", "estimated_cost", "language_pair",
        "cloud_fallback_triggered", "source_langs_in_evidence",
        "modalities_in_evidence", "escalation_triggers", "timestamp",
    }
    assert required.issubset(cols), f"Missing columns: {required - cols}"


# ── Insert / read-back tests ──────────────────────────────────────────────────

def test_log_node_basic(tmp_db):
    """log_node() inserts a row; reading it back gives the expected field values."""
    from observability.db import init_db, log_node
    init_db()
    log_node(
        query_id="q-basic",
        node_name="critic",
        estimated_cost=0.0,
        verdict="sufficient",
        groundedness_score=0.85,
        relevance_score=0.72,
    )

    conn = sqlite3.connect(str(tmp_db))
    row = conn.execute(
        "SELECT node_name, verdict, groundedness_score FROM query_log WHERE query_id='q-basic'"
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "critic"
    assert row[1] == "sufficient"
    assert abs(row[2] - 0.85) < 1e-6


def test_estimated_cost_never_null(tmp_db):
    """Every inserted row must have estimated_cost IS NOT NULL."""
    from observability.db import init_db, log_node
    init_db()
    log_node(query_id="q-notnull", node_name="reranker", estimated_cost=0.0)

    conn = sqlite3.connect(str(tmp_db))
    row = conn.execute(
        "SELECT estimated_cost FROM query_log WHERE query_id='q-notnull'"
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] is not None, "estimated_cost must never be NULL"


# ── Cost calculation tests ────────────────────────────────────────────────────

def test_gemini_cost_nonzero():
    """Gemini Flash must produce a positive cost estimate for non-zero token counts."""
    from observability.cost import compute_cost
    cost = compute_cost("gemini-2.5-flash", 1000, 500)
    assert cost > 0, f"Expected cost > 0 for gemini-2.5-flash, got {cost}"


def test_local_cost_zero():
    """DeepSeek (free tier) must always return exactly 0.0."""
    from observability.cost import compute_cost
    assert compute_cost("deepseek-v4-flash", 1000, 500) == 0.0


def test_log_node_zero_cost_explicit(tmp_db):
    """Reranker/retriever rows must store estimated_cost = 0.0 exactly."""
    from observability.db import init_db, log_node
    init_db()
    log_node(query_id="q-zero", node_name="reranker", estimated_cost=0.0)

    conn = sqlite3.connect(str(tmp_db))
    row = conn.execute(
        "SELECT estimated_cost FROM query_log WHERE query_id='q-zero'"
    ).fetchone()
    conn.close()

    assert row[0] == 0.0, f"Expected 0.0, got {row[0]}"


# ── Session grouping test ─────────────────────────────────────────────────────

def test_query_id_groups_rows(tmp_db):
    """Multiple log_node() calls with the same query_id must all appear under that session."""
    from observability.db import init_db, log_node
    init_db()
    sid = "session-abc-123"
    log_node(query_id=sid, node_name="intent_classifier", estimated_cost=0.001)
    log_node(query_id=sid, node_name="retriever",         estimated_cost=0.0)
    log_node(query_id=sid, node_name="reranker",          estimated_cost=0.0)

    conn = sqlite3.connect(str(tmp_db))
    rows = conn.execute(
        "SELECT node_name FROM query_log WHERE query_id=? ORDER BY id",
        (sid,),
    ).fetchall()
    conn.close()

    node_names = [r[0] for r in rows]
    assert len(node_names) == 3, f"Expected 3 rows, got {len(node_names)}"
    assert "intent_classifier" in node_names
    assert "retriever" in node_names
    assert "reranker" in node_names
