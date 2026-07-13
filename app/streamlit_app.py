"""
app/streamlit_app.py — Sprint 10

Streamlit demo UI for the Meridian RAG system.
Single-page layout: chat (left) + collapsible observability sidebar (right).

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env and add build/ to path before internal imports
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT))

_API_URL = "http://localhost:8000"
_DB_PATH = _ROOT / "observability" / "meridian.db"

st.set_page_config(page_title="Meridian RAG", page_icon="🧭", layout="wide")


# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .abstained-banner {
        background: #fff3cd;
        border-left: 4px solid #f0ad4e;
        padding: 0.75rem 1rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        color: #856404;
    }
    .obs-metric {
        font-size: 0.85rem;
        padding: 0.35rem 0;
        border-bottom: 1px solid #eee;
    }
    .obs-metric-label {
        color: #888;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .obs-metric-value {
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _call_api(query: str, bu: str) -> dict | None:
    try:
        r = requests.post(
            f"{_API_URL}/query",
            json={"query": query, "bu": bu, "top_k": 5},
            timeout=120,
        )
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.text[:300]}")
            return None
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {_API_URL}. Start it: `uvicorn app.main:app --port 8000`")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def _query_observability(query_id: str) -> list[dict]:
    """Fetch observability rows for a given query_id from SQLite."""
    if not _DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM query_log WHERE query_id = ? ORDER BY id", (query_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _summarize_obs(rows: list[dict]) -> dict:
    """Extract key observability metrics from query_log rows."""
    summary: dict = {
        "retrieval_round": None,
        "groundedness": None,
        "relevance": None,
        "verdict": None,
        "intent": None,
        "total_cost": 0.0,
        "models_used": [],
        "language_pair": None,
        "abstained": False,
    }
    for r in rows:
        node = r.get("node_name", "")
        if r.get("retrieval_round") is not None:
            summary["retrieval_round"] = r["retrieval_round"]
        if r.get("groundedness_score") is not None:
            summary["groundedness"] = r["groundedness_score"]
        if r.get("relevance_score") is not None:
            summary["relevance"] = r["relevance_score"]
        if r.get("verdict"):
            summary["verdict"] = r["verdict"]
        if node == "intent_classifier" and r.get("verdict"):
            summary["intent"] = r["verdict"]
        if r.get("estimated_cost"):
            summary["total_cost"] += float(r["estimated_cost"])
        if r.get("model_used") and r["model_used"] not in summary["models_used"]:
            summary["models_used"].append(r["model_used"])
        if r.get("language_pair"):
            summary["language_pair"] = r["language_pair"]
        desc = str(r.get("critic_reasoning") or "")
        if "abstain" in desc.lower() or "no-answer" in desc.lower():
            summary["abstained"] = True
    return summary


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    col_chat, col_sidebar = st.columns([3, 1])

    # ── Left panel: chat ──────────────────────────────────────────────────
    with col_chat:
        st.title("🧭 Meridian")
        st.caption("Cross-lingual enterprise RAG — EN · HI · ZH across text, image & audio")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("abstained"):
                    st.markdown(
                        '<div class="abstained-banner">'
                        '⚠ Honest no-answer — insufficient grounded evidence found'
                        '</div>',
                        unsafe_allow_html=True,
                    )

        bu = st.selectbox(
            "Business unit scope",
            ["All", "hr", "it_security", "product", "exec_comms"],
            index=0,
            label_visibility="collapsed",
        )
        bu_value = "" if bu == "All" else bu

        prompt = st.chat_input("Ask anything about Meridian Global Corp...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving..."):
                    result = _call_api(prompt, bu_value)

                if result is None:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "_API unavailable_", "abstained": False}
                    )
                else:
                    answer = result.get("answer", "")
                    sources = result.get("sources", [])
                    hits = result.get("hits", [])
                    abstained = result.get("abstained", False)

                    md = answer
                    if sources:
                        md += "\n\n**References:**\n"
                        for i, src in enumerate(sources, 1):
                            md += f"\n[{i}] {src}"
                    st.markdown(md)

                    if abstained:
                        st.markdown(
                            '<div class="abstained-banner">'
                            '⚠ Honest no-answer — insufficient grounded evidence found'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                    # Show cited chunks in expanders
                    if hits:
                        with st.expander("📎 Retrieved chunks"):
                            for h in hits:
                                st.caption(
                                    f"`{h['source_file']}` [{h['bu']}/{h['lang']}/{h['modality']}] "
                                    f"— score: {h['score']:.4f}"
                                )

                    query_id = result.get("query", "")  # fallback; actual query_id is internal
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": md,
                        "abstained": abstained,
                        "query_id": query_id,
                    })

                    # Store last query_id for sidebar. Use the session's query_id from the API
                    # The API response doesn't expose query_id directly, so we store the answer
                    # hash as a key and look up observability by recency.
                    st.session_state.last_query = prompt

    # ── Right panel: observability ────────────────────────────────────────
    with col_sidebar:
        st.header("📊 Observability")

        # Look up the most recent observability row batch
        if st.session_state.get("last_query"):
            # Find the most recent rows by timestamp
            obs_rows = _query_observability("")  # empty query_id means nothing
            # Instead, pick the most recent batch of rows
            if _DB_PATH.exists():
                conn = sqlite3.connect(str(_DB_PATH))
                conn.row_factory = sqlite3.Row
                latest = conn.execute(
                    "SELECT query_id FROM query_log ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if latest:
                    obs_rows = _query_observability(latest["query_id"])
                conn.close()

            if obs_rows:
                s = _summarize_obs(obs_rows)

                # Intent
                st.markdown('<div class="obs-metric">', unsafe_allow_html=True)
                st.markdown('<div class="obs-metric-label">Intent</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="obs-metric-value">{s["intent"] or "—"}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # Retrieval round
                st.markdown('<div class="obs-metric">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="obs-metric-label">Retrieval round</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="obs-metric-value">{s["retrieval_round"] or 0}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # Groundedness
                g_color = "#28a745" if (s["groundedness"] or 0) >= 0.7 else "#dc3545"
                g_text = f"{s['groundedness']:.2f}" if s["groundedness"] is not None else "—"
                st.markdown('<div class="obs-metric">', unsafe_allow_html=True)
                st.markdown('<div class="obs-metric-label">Groundedness</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="obs-metric-value" style="color:{g_color}">{g_text}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # Relevance
                r_color = "#28a745" if (s["relevance"] or 0) >= 0.6 else "#dc3545"
                r_text = f"{s['relevance']:.2f}" if s["relevance"] is not None else "—"
                st.markdown('<div class="obs-metric">', unsafe_allow_html=True)
                st.markdown('<div class="obs-metric-label">Relevance</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="obs-metric-value" style="color:{r_color}">{r_text}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # Verdict
                if s["verdict"]:
                    st.markdown('<div class="obs-metric">', unsafe_allow_html=True)
                    st.markdown('<div class="obs-metric-label">Critic verdict</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="obs-metric-value">{s["verdict"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                # Language pair
                if s["language_pair"]:
                    st.markdown('<div class="obs-metric">', unsafe_allow_html=True)
                    st.markdown('<div class="obs-metric-label">Language pair</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="obs-metric-value">{s["language_pair"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                # Cost
                st.markdown('<div class="obs-metric">', unsafe_allow_html=True)
                st.markdown('<div class="obs-metric-label">Est. cost</div>', unsafe_allow_html=True)
                cost_str = f"${s['total_cost']:.6f}" if s["total_cost"] > 0 else "$0.00"
                st.markdown(
                    f'<div class="obs-metric-value">{cost_str}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # Models used
                if s["models_used"]:
                    st.markdown('<div class="obs-metric">', unsafe_allow_html=True)
                    st.markdown('<div class="obs-metric-label">Models</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="obs-metric-value">{", ".join(s["models_used"])}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                # Abstained warning
                if s["abstained"]:
                    st.markdown(
                        '<div class="abstained-banner" style="margin-top:1rem">'
                        '⚠ Abstained — insufficient evidence'
                        '</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No observability data yet. Run a query first.")
        else:
            st.caption("Observability metrics appear here after each query.")


if __name__ == "__main__":
    main()
