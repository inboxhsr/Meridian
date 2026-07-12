"""
app/models.py — Sprint 4

Pydantic request/response models for the Meridian REST API.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Question to ask")
    bu: str = Field(
        default="",
        pattern=r"^(hr|it_security|product|exec_comms|)$",
        description="Optional business-unit scope filter",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Max chunks to retrieve")
    skip_pii: bool = Field(default=False, description="Skip PII guard (use in testing)")

    model_config = {"json_schema_extra": {
        "example": {
            "query": "What is the travel expense reimbursement limit?",
            "bu": "hr",
            "top_k": 5,
            "skip_pii": False,
        }
    }}


# ── Response ──────────────────────────────────────────────────────────────────

class ChunkHit(BaseModel):
    source_file: str
    bu: str
    lang: str
    modality: str
    score: float


class QueryResponse(BaseModel):
    query: str                  = Field(description="Original query")
    safe_query: str             = Field(description="PII-redacted query used for retrieval")
    lang: str                   = Field(description="Detected language: en | hi | zh")
    pii_flagged: bool           = Field(description="Whether PII was detected in the query")
    answer: str                 = Field(description="Grounded answer from DeepSeek")
    sources: list[str]          = Field(description="Unique source filenames cited")
    chunks_used: int            = Field(description="Context chunks included in the prompt")
    hits: list[ChunkHit]        = Field(description="All retrieved Milvus hits")


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str          = Field(description="'ok' or 'degraded'")
    milvus: bool         = Field(description="Milvus reachable")
    corpus_chunks: int   = Field(description="Total chunks in meridian_corpus")
    version: str         = Field(description="API version")
