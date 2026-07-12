"""
app/routes/query.py — Sprint 4

POST /query — full RAG pipeline endpoint.

Flow: route → retrieve → generate → respond
"""

import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models import QueryRequest, QueryResponse, ChunkHit
from pipeline.router    import route
from pipeline.retriever import retrieve
from pipeline.generator import generate

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against the Meridian knowledge base",
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Runs the full RAG pipeline:

    1. **Route** — detect language, check PII, apply BU filter
    2. **Retrieve** — embed query, ANN search Milvus top-K chunks
    3. **Generate** — build grounded prompt, call DeepSeek, return cited answer

    Audio chunks (MP3 transcriptions) are included after Sprint 4 re-ingest.
    """
    try:
        # 1. Route
        routing = route(
            request.query,
            bu_filter=request.bu,
            skip_pii=request.skip_pii,
        )

        # 2. Retrieve
        chunks = retrieve(
            routing["safe_query"],
            top_k=request.top_k,
            bu_filter=routing["bu_filter"],
        )

        # 3. Generate
        result = generate(routing["safe_query"], chunks, lang=routing["lang"])

        return QueryResponse(
            query=routing["query"],
            safe_query=routing["safe_query"],
            lang=routing["lang"],
            pii_flagged=routing["pii_flagged"],
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"],
            hits=[
                ChunkHit(
                    source_file=c["source_file"],
                    bu=c["bu"],
                    lang=c["lang"],
                    modality=c["modality"],
                    score=round(c["score"], 6),
                )
                for c in chunks
            ],
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
