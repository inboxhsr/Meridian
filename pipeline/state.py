"""
pipeline/state.py — Sprint 5

LangGraph TypedDict state schema for the Meridian agentic pipeline.
Every node reads from and writes to this shared state dict.
"""

from __future__ import annotations
from typing import TypedDict


class MeridianState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str            # original user query (verbatim)
    bu_filter: str        # business unit filter; '' = all BUs
    top_k: int            # number of chunks to retrieve per sub-question

    # ── Routing / language ────────────────────────────────────────────────────
    lang: str             # detected language: 'en' | 'hi' | 'zh'
    intent: str           # 'simple' | 'multi_hop' | 'out_of_scope'
    safe_query: str       # PII-redacted version of query
    pii_flagged: bool     # True if PII was detected and redacted

    # ── Query rewriting ───────────────────────────────────────────────────────
    sub_questions: list[str]   # [] → use safe_query directly; N>1 → compound

    # ── Retrieval ─────────────────────────────────────────────────────────────
    chunks: list[dict]         # current retrieved + deduplicated chunks
    retrieval_round: int       # how many retrieval attempts have occurred (starts at 0)

    # ── Critic ────────────────────────────────────────────────────────────────
    groundedness_score: float  # 0.0–1.0; threshold ≥ 0.7 for 'sufficient'
    relevance_score: float     # 0.0–1.0; threshold ≥ 0.6 for 'sufficient'
    verdict: str               # 'sufficient' | 'retry' | 'abstain'
    critic_reasoning: str      # one-sentence explanation (logged; never used programmatically)

    # ── Output ────────────────────────────────────────────────────────────────
    answer: str           # final answer text
    sources: list[str]    # unique source filenames cited
    chunks_used: int      # number of context chunks passed to the generator
    abstained: bool       # True if the pipeline gave an honest no-answer
