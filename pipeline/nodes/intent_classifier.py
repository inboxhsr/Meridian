"""
pipeline/nodes/intent_classifier.py — Sprint 5

Classifies the user query into one of three intents using Gemini Flash:
  - 'simple'       single factual lookup (expense limit, policy date, etc.)
  - 'multi_hop'    compound/comparative query requiring multiple sources
  - 'out_of_scope' unrelated to Meridian knowledge base topics

Also absorbs language detection and PII guard by calling pipeline.router.route().

Safe-default on parse failure: 'multi_hop' — conservative, never silently
drops a query to out_of_scope.
"""

from __future__ import annotations
import json
import os

from google import genai

from pipeline.router import route as _route
from pipeline.state import MeridianState

_VALID_INTENTS = {"simple", "multi_hop", "out_of_scope"}
_DEFAULT_INTENT = "multi_hop"

_SYSTEM_PROMPT = (
    "You are a query intent classifier for Meridian Global Corp's enterprise knowledge base.\n"
    "The knowledge base covers: HR policies, IT Security policies, Product specs, and Executive Comms.\n\n"
    "Classify the query into EXACTLY one of these intents:\n"
    "  simple      - single factual lookup (e.g. expense limit, policy date, one topic)\n"
    "  multi_hop   - compound or comparative query needing multiple sources or reasoning steps\n"
    "  out_of_scope - completely unrelated to HR, IT Security, Product, or Executive Comms\n\n"
    'Respond ONLY with a JSON object, no explanation: {"intent": "simple"|"multi_hop"|"out_of_scope"}'
)


def classify_intent(state: MeridianState) -> dict:
    """LangGraph node — classify query intent, detect language, check PII.

    Reads:  query, bu_filter, top_k  (from initial state)
    Writes: lang, intent, safe_query, pii_flagged
    """
    query = state["query"]
    bu_filter = state.get("bu_filter", "")

    # ── Language detection + PII guard (reuse Sprint 3 router) ───────────────
    routed = _route(query, bu_filter=bu_filter, skip_pii=False)
    safe_query = routed["safe_query"]
    lang = routed["lang"]
    pii_flagged = routed["pii_flagged"]

    # ── Intent classification via Gemini Flash ────────────────────────────────
    intent = _classify(safe_query)

    return {
        "lang": lang,
        "intent": intent,
        "safe_query": safe_query,
        "pii_flagged": pii_flagged,
    }


def _classify(safe_query: str) -> str:
    """Call Gemini Flash to classify the intent. Return safe-default on failure."""
    if not safe_query or not safe_query.strip():
        return "out_of_scope"

    key = os.environ.get("GEMINI_API_KEY_A", "")
    if not key:
        return _DEFAULT_INTENT

    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {"role": "user", "parts": [{"text": _SYSTEM_PROMPT + "\n\nQuery: " + safe_query}]}
            ],
        )
        raw = (resp.text or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        intent = data.get("intent", _DEFAULT_INTENT)
        return intent if intent in _VALID_INTENTS else _DEFAULT_INTENT
    except Exception:
        return _DEFAULT_INTENT
