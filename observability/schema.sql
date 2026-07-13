-- observability/schema.sql — Sprint 8
-- Human-readable DDL. Mirrors observability/db.py exactly.
-- The actual table is created by init_db() in db.py at runtime.

CREATE TABLE IF NOT EXISTS query_log (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id                 TEXT    NOT NULL,   -- UUID per graph invocation
    node_name                TEXT    NOT NULL,   -- e.g. 'intent_classifier', 'critic'
    retrieval_round          INTEGER,            -- NULL for non-retrieval nodes
    model_used               TEXT,               -- e.g. 'gemini-2.5-flash', NULL for local
    groundedness_score       REAL,               -- 0.0–1.0; populated by critic only
    relevance_score          REAL,               -- 0.0–1.0; populated by critic only
    verdict                  TEXT,               -- 'sufficient' | 'retry' | 'abstain'
    critic_reasoning         TEXT,               -- one-sentence explanation (logged only)
    tokens_used              INTEGER,            -- input+output tokens (or chunk count proxy)
    estimated_cost           REAL    NOT NULL,   -- USD; 0.0 for local/free-tier, NEVER NULL
    language_pair            TEXT,               -- e.g. 'hi->zh' for cross-lingual queries
    cloud_fallback_triggered INTEGER,            -- 0 or 1
    source_langs_in_evidence TEXT,              -- JSON array e.g. '["en","zh"]'
    modalities_in_evidence   TEXT,              -- JSON array e.g. '["text","audio"]'
    escalation_triggers      TEXT,              -- JSON array of which routing rules fired
    timestamp                TEXT    NOT NULL    -- ISO 8601 UTC timestamp
);
