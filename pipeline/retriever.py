"""
pipeline/retriever.py — Sprint 3

Embeds a query and retrieves the top-K most semantically similar chunks from Milvus.

Key: query embeddings use task_type='RETRIEVAL_QUERY' (vs 'RETRIEVAL_DOCUMENT'
used during indexing) — this is the asymmetric embedding pattern recommended by
Google for optimal Inner Product retrieval.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import embedder as emb
from ingest import milvus_store as store


def retrieve(
    query: str,
    top_k: int = 5,
    bu_filter: str = "",
) -> list[dict]:
    """Embed query and retrieve top-K chunks from Milvus.

    Args:
        query:      Natural language question (already PII-safe)
        top_k:      Maximum number of chunks to return
        bu_filter:  Optional BU name to restrict search scope

    Returns:
        List of hit dicts: {score, text, source_file, bu, lang, modality, chunk_index}
    """
    # 1. Embed the query
    emb_client = emb.get_client()
    query_vec = emb.embed_one(emb_client, query, task_type="RETRIEVAL_QUERY")

    # 2. Build optional Milvus filter expression
    filter_expr = f'bu == "{bu_filter}"' if bu_filter else ""

    # 3. ANN search
    mv_client = store.get_client()
    results = store.search(mv_client, query_vec, top_k=top_k, filter_expr=filter_expr)

    return results
