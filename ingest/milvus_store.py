"""
ingest/milvus_store.py — Sprint 2

Milvus collection management using the MilvusClient API (pymilvus >= 2.4).

Collection: meridian_corpus
Schema:
  id           INT64  (primary, auto_id)
  source_file  VARCHAR(255)   — e.g. "hr_expense_policy.en.pdf"
  bu           VARCHAR(50)    — hr | it_security | product | exec_comms
  lang         VARCHAR(10)    — en | hi | zh
  modality     VARCHAR(20)    — pdf | slide | audio
  chunk_index  INT64          — 0-based chunk position within the document
  text         VARCHAR(65535) — raw chunk text
  embedding    FLOAT_VECTOR(768)

Index: IVF_FLAT, Inner Product (cosine equiv after normalisation)
"""

import os
from pymilvus import MilvusClient, DataType

COLLECTION = "meridian_corpus"
DIM = 768

# ── Connection ────────────────────────────────────────────────────────────────

def get_client() -> MilvusClient:
    host = os.environ.get("MILVUS_HOST", "localhost")
    port = os.environ.get("MILVUS_PORT", "19530")
    return MilvusClient(uri=f"http://{host}:{port}")


# ── Schema / collection lifecycle ────────────────────────────────────────────

def ensure_collection(client: MilvusClient, *, reset: bool = False) -> None:
    """Create the collection (and IVF_FLAT index) if it doesn't exist.

    Args:
        reset: If True, drop and recreate even if it exists.
    """
    if reset and client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)

    if client.has_collection(COLLECTION):
        return  # already exists, nothing to do

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id",          DataType.INT64,        is_primary=True)
    schema.add_field("source_file", DataType.VARCHAR,      max_length=255)
    schema.add_field("bu",          DataType.VARCHAR,      max_length=50)
    schema.add_field("lang",        DataType.VARCHAR,      max_length=10)
    schema.add_field("modality",    DataType.VARCHAR,      max_length=20)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("text",        DataType.VARCHAR,      max_length=65535)
    schema.add_field("embedding",   DataType.FLOAT_VECTOR, dim=DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="IP",       # inner product ≈ cosine for normalised vectors
        params={"nlist": 64},
    )

    client.create_collection(
        COLLECTION,
        schema=schema,
        index_params=index_params,
    )


# ── Insert ────────────────────────────────────────────────────────────────────

BATCH_SIZE = 50  # insert in batches to stay within Milvus gRPC message limits


def insert_records(client: MilvusClient, records: list[dict]) -> int:
    """Insert embedded chunk records. Returns count of rows inserted."""
    if not records:
        return 0

    total = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        # Truncate text to VARCHAR limit (leave 35 chars of headroom)
        safe = [
            {**r, "text": r["text"][:65000]}
            for r in batch
        ]
        client.insert(collection_name=COLLECTION, data=safe)
        total += len(batch)

    return total


# ── Query helpers ─────────────────────────────────────────────────────────────

def count(client: MilvusClient) -> int:
    """Return total number of records in the collection."""
    if not client.has_collection(COLLECTION):
        return 0
    res = client.query(COLLECTION, filter="", output_fields=["count(*)"])
    return res[0]["count(*)"] if res else 0


def already_ingested(client: MilvusClient, source_file: str) -> bool:
    """Return True if at least one chunk for source_file is already in Milvus."""
    if not client.has_collection(COLLECTION):
        return False
    res = client.query(
        COLLECTION,
        filter=f'source_file == "{source_file}"',
        output_fields=["id"],
        limit=1,
    )
    return len(res) > 0


def search(
    client: MilvusClient,
    query_vector: list[float],
    top_k: int = 5,
    filter_expr: str = "",
) -> list[dict]:
    """ANN search. Returns list of hit dicts with text, source_file, lang, score."""
    results = client.search(
        collection_name=COLLECTION,
        data=[query_vector],
        limit=top_k,
        filter=filter_expr,
        output_fields=["text", "source_file", "bu", "lang", "modality", "chunk_index"],
    )
    hits = []
    for hit in results[0]:
        hits.append({
            "score":       hit["distance"],
            "text":        hit["entity"]["text"],
            "source_file": hit["entity"]["source_file"],
            "bu":          hit["entity"]["bu"],
            "lang":        hit["entity"]["lang"],
            "modality":    hit["entity"]["modality"],
            "chunk_index": hit["entity"]["chunk_index"],
        })
    return hits
