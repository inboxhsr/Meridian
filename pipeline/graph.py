"""
pipeline/graph.py — Sprint 6

Assembles the full Meridian LangGraph agentic pipeline.

Graph topology:
  intent_classifier
       ├── out_of_scope → abstainer → END
       ├── simple       → retriever
       └── multi_hop    → query_rewriter → retriever
                                               ↓
                                           reranker  ← Sprint 6
                                               ↓
                                            critic
                                       ├── sufficient → generator → END
                                       ├── retry      → query_rewriter (loop)
                                       └── abstain    → abstainer → END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from pipeline.state import MeridianState
from pipeline.nodes.intent_classifier import classify_intent
from pipeline.nodes.query_rewriter import rewrite_query
from pipeline.nodes.retriever import retrieve
from pipeline.nodes.reranker import rerank
from pipeline.nodes.critic import grade_context
from pipeline.nodes.generator import generate_answer
from pipeline.nodes.abstainer import abstain


# ── Conditional edge functions ────────────────────────────────────────────────

def _route_after_intent(state: MeridianState) -> str:
    """Route based on classified intent."""
    intent = state.get("intent", "multi_hop")
    if intent == "out_of_scope":
        return "abstainer"
    if intent == "multi_hop":
        return "query_rewriter"
    return "retriever"  # simple


def _route_after_critic(state: MeridianState) -> str:
    """Route based on critic verdict."""
    verdict = state.get("verdict", "retry")
    if verdict == "sufficient":
        return "generator"
    if verdict == "retry":
        return "query_rewriter"
    return "abstainer"  # abstain or exhausted rounds


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph():
    """Build and compile the Meridian LangGraph pipeline."""
    g = StateGraph(MeridianState)

    # Register nodes
    g.add_node("intent_classifier", classify_intent)
    g.add_node("query_rewriter",    rewrite_query)
    g.add_node("retriever",         retrieve)
    g.add_node("reranker",          rerank)
    g.add_node("critic",            grade_context)
    g.add_node("generator",         generate_answer)
    g.add_node("abstainer",         abstain)

    # Entry point
    g.set_entry_point("intent_classifier")

    # Edges from intent_classifier (conditional)
    g.add_conditional_edges(
        "intent_classifier",
        _route_after_intent,
        {
            "query_rewriter": "query_rewriter",
            "retriever":      "retriever",
            "abstainer":      "abstainer",
        },
    )

    # Fixed edges
    g.add_edge("query_rewriter", "retriever")
    g.add_edge("retriever",      "reranker")
    g.add_edge("reranker",       "critic")

    # Edges from critic (conditional — CRAG loop)
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "generator":      "generator",
            "query_rewriter": "query_rewriter",
            "abstainer":      "abstainer",
        },
    )

    # Terminal edges
    g.add_edge("generator", END)
    g.add_edge("abstainer", END)

    return g.compile()


# Module-level compiled graph — import and call graph.invoke(initial_state)
graph = build_graph()
