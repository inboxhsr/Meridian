"""
app/main.py — Sprint 4

FastAPI application entry point for the Meridian RAG API.

Run locally:
    uvicorn app.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env before any pipeline imports
load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.routes import health as health_routes
from app.routes import query as query_routes


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm Milvus connection and embedding client on startup."""
    try:
        from ingest import milvus_store as store
        from ingest import embedder as emb
        app.state.mv_client  = store.get_client()
        app.state.emb_client = emb.get_client()
        print(f"[startup] Milvus OK — {store.count(app.state.mv_client)} chunks indexed")
    except Exception as exc:
        print(f"[startup] WARNING: {exc}")
    yield
    # No teardown needed for Milvus HTTP client


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Meridian RAG API",
    description=(
        "Enterprise knowledge assistant for Meridian Global Corp. "
        "Supports multilingual queries (EN / HI / ZH) across HR, IT Security, "
        "Product, and Executive Communications corpora."
    ),
    version="0.4.0",
    lifespan=lifespan,
)

# Allow all origins in dev — tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_routes.router, tags=["Health"])
app.include_router(query_routes.router,  tags=["Query"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Meridian RAG API",
        "version": "0.4.0",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query",
    }
