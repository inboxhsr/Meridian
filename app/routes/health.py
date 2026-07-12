"""
app/routes/health.py — Sprint 4

GET /health — returns Milvus connectivity and corpus chunk count.
"""

from fastapi import APIRouter
from app.models import HealthResponse

router = APIRouter()

API_VERSION = "0.4.0"


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health() -> HealthResponse:
    """Returns service status and corpus stats."""
    try:
        from ingest import milvus_store as store
        client = store.get_client()
        n = store.count(client)
        milvus_ok = True
    except Exception:
        n = 0
        milvus_ok = False

    return HealthResponse(
        status="ok" if milvus_ok else "degraded",
        milvus=milvus_ok,
        corpus_chunks=n,
        version=API_VERSION,
    )
