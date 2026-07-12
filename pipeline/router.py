"""
pipeline/router.py — Sprint 3

Routes an incoming query through:
  1. Language detection  (fast heuristic, no API call)
  2. PII guard           (two-stage: regex pre-filter → Gemini Flash confirmation)
  3. BU filter tagging   (passed in or parsed from explicit CLI arg)

The two-stage PII approach preserves Gemini's 20 req/day free-tier quota —
Gemini is only called when the fast regex pre-filter finds a likely PII pattern.
"""

from __future__ import annotations
import os
import re

# ── Language detection ────────────────────────────────────────────────────────

_CJK_RE  = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
_DEVA_RE = re.compile(r"[\u0900-\u097f]")


def detect_language(text: str) -> str:
    """Heuristic language detection from character script.

    Returns 'zh', 'hi', or 'en'.  No API call needed.
    """
    if not text:
        return "en"
    cjk  = len(_CJK_RE.findall(text))
    deva = len(_DEVA_RE.findall(text))
    total = max(len(text), 1)
    if cjk  / total > 0.08:
        return "zh"
    if deva / total > 0.08:
        return "hi"
    return "en"


# ── PII guard ─────────────────────────────────────────────────────────────────

# Stage 1 — fast regex patterns (no API call)
_PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),   # email
    re.compile(r"\b\d{3}[-.\s]\d{2}[-.\s]\d{4}\b"),                        # SSN
    re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"),                        # US phone
    re.compile(r"\b(?:\d[ -]?){13,16}\b"),                                  # credit card
    re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),                                   # passport
]


def _regex_pii_check(text: str) -> bool:
    """Return True if any PII-like pattern is found."""
    return any(p.search(text) for p in _PII_PATTERNS)


def check_pii(query: str) -> dict:
    """Two-stage PII check. Returns {has_pii, redacted}.

    Stage 1: fast regex — if no pattern found, return immediately (no API call).
    Stage 2: Gemini Flash confirmation + redaction (only if Stage 1 triggers).
    """
    if not _regex_pii_check(query):
        return {"has_pii": False, "redacted": query}

    # Stage 2 — Gemini Flash (only reached when regex fires)
    key = os.environ.get("GEMINI_API_KEY_A", "")
    if not key:
        # Fail-open: pass query through, flag as PII suspected
        return {"has_pii": True, "redacted": "[REDACTED — PII suspected]"}

    try:
        from google import genai
        client = genai.Client(api_key=key)
        prompt = (
            "Analyze this query for Personally Identifiable Information (PII).\n"
            "PII includes: email addresses, phone numbers, SSNs, passport numbers, "
            "credit card numbers, home addresses.\n\n"
            f"Query: {query}\n\n"
            "Respond in EXACTLY this format (two lines only):\n"
            "HAS_PII: yes/no\n"
            "REDACTED: <query with PII replaced by [REDACTED], or original if no PII>"
        )
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        lines = {}
        for line in (resp.text or "").strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                lines[k.strip()] = v.strip()
        has_pii = lines.get("HAS_PII", "no").lower() == "yes"
        redacted = lines.get("REDACTED", query)
        return {"has_pii": has_pii, "redacted": redacted}
    except Exception:
        # Fail-open on Gemini errors
        return {"has_pii": True, "redacted": "[REDACTED — PII suspected]"}


# ── Main router ───────────────────────────────────────────────────────────────

def route(query: str, bu_filter: str = "", skip_pii: bool = False) -> dict:
    """Route an incoming query.

    Returns:
        query       : str   — original query
        safe_query  : str   — PII-redacted version (same as query if no PII)
        lang        : str   — detected language: 'en' | 'hi' | 'zh'
        bu_filter   : str   — BU to scope the Milvus search ('' = all)
        pii_flagged : bool  — True if PII was detected and redacted
    """
    lang = detect_language(query)
    pii  = {"has_pii": False, "redacted": query} if skip_pii else check_pii(query)

    return {
        "query":       query,
        "safe_query":  pii["redacted"],
        "lang":        lang,
        "bu_filter":   bu_filter,
        "pii_flagged": pii["has_pii"],
    }
