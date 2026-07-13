"""
app/routes/query.py — Sprint 5 / Sprint 8 (observability)

POST /query — LangGraph agentic pipeline endpoint.

Flow: intent_classifier → [query_rewriter] → retriever → critic → [loop] → generator | abstainer
"""

import sys
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models import QueryRequest, QueryResponse, ChunkHit
from observability.db import init_db
from pipeline.graph import graph

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against the Meridian knowledge base",
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Runs the full LangGraph agentic RAG pipeline:

    1. **Intent Classifier** — detect language, check PII, classify intent
    2. **Query Rewriter** — decompose compound queries into sub-questions (multi_hop only)
    3. **Retriever** — embed + ANN search Milvus, deduplicate across sub-questions
    4. **Critic** — grade retrieved context; retry or abstain if insufficient
    5. **Generator** — grounded answer with citations (DeepSeek)
       OR **Abstainer** — honest no-answer after max rounds or out-of-scope
    """
    try:
        session_id = str(uuid.uuid4())
        try:
            init_db()
        except Exception:
            pass

        initial_state = {
            "query_id":           session_id,
            "query":              request.query,
            "bu_filter":          request.bu,
            "top_k":              request.top_k,
            # Fields populated by nodes — provide defaults so TypedDict is satisfied
            "lang":               "en",
            "intent":             "",
            "safe_query":         request.query,
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

        result = graph.invoke(initial_state)

        chunks = result.get("chunks", [])
        top_k = request.top_k

        return QueryResponse(
            query=result.get("query", request.query),
            safe_query=result.get("safe_query", request.query),
            lang=result.get("lang", "en"),
            pii_flagged=result.get("pii_flagged", False),
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            chunks_used=result.get("chunks_used", 0),
            hits=[
                ChunkHit(
                    source_file=c["source_file"],
                    bu=c["bu"],
                    lang=c["lang"],
                    modality=c["modality"],
                    score=round(c["score"], 6),
                )
                for c in chunks[:top_k]
            ],
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
