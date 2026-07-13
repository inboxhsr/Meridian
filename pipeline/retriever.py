"""
pipeline/retriever.py — Sprint 7

Embeds a query and retrieves the top-K most relevant chunks from Milvus
using hybrid dense + sparse (BM25) search with RRF fusion.

Key: query embeddings use task_type='RETRIEVAL_QUERY' (vs 'RETRIEVAL_DOCUMENT'
used during indexing) — this is the asymmetric embedding pattern recommended by
Google for optimal Inner Product retrieval.

Sprint 7 change: search() → hybrid_search() (dense + BM25 sparse, RRF fusion).
Public signature is unchanged.
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
    """Embed query and retrieve top-K chunks via hybrid dense + sparse search.

    Args:
        query:      Natural language question (already PII-safe)
        top_k:      Maximum number of chunks to return
        bu_filter:  Optional BU name to restrict search scope

    Returns:
        List of hit dicts: {score, text, source_file, bu, lang, modality, chunk_index}
        score is the RRF-fused rank score from Milvus WeightedRanker(0.5, 0.5).
    """
    # 1. Embed the query (dense)
    emb_client = emb.get_client()
    query_vec = emb.embed_one(emb_client, query, task_type="RETRIEVAL_QUERY")

    # 2. Build optional Milvus filter expression
    filter_expr = f'bu == "{bu_filter}"' if bu_filter else ""

    # 3. Hybrid search — dense + BM25 sparse, RRF fusion
    mv_client = store.get_client()
    results = store.hybrid_search(
        mv_client,
        query_vector=query_vec,
        query_text=query,
        top_k=top_k,
        filter_expr=filter_expr,
    )

    return results
