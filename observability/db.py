"""
observability/db.py — Sprint 8

SQLite schema creation and insert helper.
All pipeline nodes call log_node() to record one row per call.
The DB file lives at observability/meridian.db (git-ignored, runtime artifact).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(__file__).parent / "meridian.db"


def _get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite DB, creating the file if needed."""
    return sqlite3.connect(str(_DB_PATH))


def init_db() -> None:
    """Create the query_log table if it doesn't exist. Idempotent."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id                 TEXT    NOT NULL,
            node_name                TEXT    NOT NULL,
            retrieval_round          INTEGER,
            model_used               TEXT,
            groundedness_score       REAL,
            relevance_score          REAL,
            verdict                  TEXT,
            critic_reasoning         TEXT,
            tokens_used              INTEGER,
            estimated_cost           REAL    NOT NULL,
            language_pair            TEXT,
            cloud_fallback_triggered INTEGER,
            source_langs_in_evidence TEXT,
            modalities_in_evidence   TEXT,
            escalation_triggers      TEXT,
            timestamp                TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_node(
    *,
    query_id: str,
    node_name: str,
    estimated_cost: float,
    retrieval_round: int | None = None,
    model_used: str | None = None,
    groundedness_score: float | None = None,
    relevance_score: float | None = None,
    verdict: str | None = None,
    critic_reasoning: str | None = None,
    tokens_used: int | None = None,
    language_pair: str | None = None,
    cloud_fallback_triggered: bool = False,
    source_langs_in_evidence: list[str] | None = None,
    modalities_in_evidence: list[str] | None = None,
    escalation_triggers: list[str] | None = None,
) -> None:
    """Insert one observability row. estimated_cost must never be None."""
    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO query_log (
            query_id, node_name, retrieval_round, model_used,
            groundedness_score, relevance_score, verdict, critic_reasoning,
            tokens_used, estimated_cost, language_pair,
            cloud_fallback_triggered, source_langs_in_evidence,
            modalities_in_evidence, escalation_triggers, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            query_id,
            node_name,
            retrieval_round,
            model_used,
            groundedness_score,
            relevance_score,
            verdict,
            critic_reasoning,
            tokens_used,
            float(estimated_cost),          # enforce float, never NULL
            language_pair,
            1 if cloud_fallback_triggered else 0,
            json.dumps(source_langs_in_evidence) if source_langs_in_evidence else None,
            json.dumps(modalities_in_evidence) if modalities_in_evidence else None,
            json.dumps(escalation_triggers) if escalation_triggers else None,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()
