"""
ingest/milvus_store.py — Sprint 7

Milvus collection management using the MilvusClient API (pymilvus >= 2.4.4).

Collection: meridian_corpus
Schema:
  id           INT64  (primary, auto_id)
  source_file  VARCHAR(255)   — e.g. "hr_expense_policy.en.pdf"
  bu           VARCHAR(50)    — hr | it_security | product | exec_comms
  lang         VARCHAR(10)    — en | hi | zh
  modality     VARCHAR(20)    — pdf | slide | audio
  chunk_index  INT64          — 0-based chunk position within the document
  text         VARCHAR(65535) — raw chunk text (also BM25 source field)
  embedding    FLOAT_VECTOR(3072)       — dense, Gemini gemini-embedding-001
  sparse_vector SPARSE_FLOAT_VECTOR     — BM25 auto-generated from 'text'

Indexes:
  embedding:     IVF_FLAT, Inner Product (cosine equiv after normalisation)
  sparse_vector: SPARSE_INVERTED_INDEX, BM25

Sprint 7 change: added sparse_vector + BM25 function + hybrid_search().
Existing search() is preserved for test backward-compatibility.
"""

import os
from pymilvus import MilvusClient, DataType, Function, FunctionType

COLLECTION = "meridian_corpus"
DIM = 3072  # gemini-embedding-001 output dimension

# ── Connection ────────────────────────────────────────────────────────────────

def get_client() -> MilvusClient:
    host = os.environ.get("MILVUS_HOST", "localhost")
    port = os.environ.get("MILVUS_PORT", "19530")
    return MilvusClient(uri=f"http://{host}:{port}")


# ── Schema / collection lifecycle ────────────────────────────────────────────

def ensure_collection(client: MilvusClient, *, reset: bool = False) -> None:
    """Create the collection (dense + sparse indexes) if it doesn't exist.

    Args:
        reset: If True, drop and recreate even if it exists.
    """
    if reset and client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)

    if client.has_collection(COLLECTION):
        return  # already exists, nothing to do

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id",            DataType.INT64,               is_primary=True)
    schema.add_field("source_file",   DataType.VARCHAR,             max_length=255)
    schema.add_field("bu",            DataType.VARCHAR,             max_length=50)
    schema.add_field("lang",          DataType.VARCHAR,             max_length=10)
    schema.add_field("modality",      DataType.VARCHAR,             max_length=20)
    schema.add_field("chunk_index",   DataType.INT64)
    schema.add_field("text",          DataType.VARCHAR,             max_length=65535,
                     enable_analyzer=True)                          # required for BM25
    schema.add_field("embedding",     DataType.FLOAT_VECTOR,        dim=DIM)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    # BM25 function: Milvus auto-generates sparse_vector from text at insert time
    bm25_fn = Function(
        name="text_bm25",
        input_field_names=["text"],
        output_field_names=["sparse_vector"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_fn)

    index_params = client.prepare_index_params()

    # Dense index
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="IP",       # inner product ≈ cosine for normalised vectors
        params={"nlist": 64},
    )

    # Sparse BM25 index
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )

    client.create_collection(
        COLLECTION,
        schema=schema,
        index_params=index_params,
    )


# ── Insert ────────────────────────────────────────────────────────────────────

BATCH_SIZE = 50  # insert in batches to stay within Milvus gRPC message limits


def insert_records(client: MilvusClient, records: list[dict]) -> int:
    """Insert embedded chunk records. Returns count of rows inserted.

    Note: do NOT include 'sparse_vector' in records — Milvus generates it
    automatically from the 'text' field via the BM25 function.
    """
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
    """Dense-only ANN search (preserved for backward compatibility with tests).

    Prefer hybrid_search() for production queries.
    """
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


def hybrid_search(
    client: MilvusClient,
    query_vector: list[float],
    query_text: str,
    top_k: int = 5,
    filter_expr: str = "",
) -> list[dict]:
    """Hybrid dense + sparse (BM25) search with RRF fusion.

    Dense and sparse results are combined using WeightedRanker(0.5, 0.5) —
    equal weight to semantic and exact-term signals. Sprint 9 RAGAS eval
    will quantify the optimal split.

    Args:
        query_vector:  Dense embedding of the query (3072-dim, RETRIEVAL_QUERY)
        query_text:    Raw query text for BM25 sparse matching
        top_k:         Max results to return after fusion
        filter_expr:   Optional Milvus filter expression (e.g. 'bu == "hr"')
    """
    from pymilvus import AnnSearchRequest, WeightedRanker

    dense_req = AnnSearchRequest(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        expr=filter_expr or None,
    )
    sparse_req = AnnSearchRequest(
        data=[query_text],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=top_k,
        expr=filter_expr or None,
    )

    results = client.hybrid_search(
        collection_name=COLLECTION,
        reqs=[dense_req, sparse_req],
        ranker=WeightedRanker(0.5, 0.5),
        limit=top_k,
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
