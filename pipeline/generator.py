"""
pipeline/generator.py — Sprint 3

Builds a grounded answer from retrieved chunks using DeepSeek.

Grounding rules:
  - Answer drawn ONLY from provided context chunks
  - Cite source documents by filename in [brackets]
  - Answer in the detected language (en / hi / zh)
  - If answer not in context, say so explicitly — no hallucination
"""

from __future__ import annotations
import os
from openai import OpenAI

DEEPSEEK_MODEL = "deepseek-v4-flash"
MAX_CONTEXT_CHARS = 6_000  # conservative context budget

_LANG_INSTRUCTION = {
    "en": "Answer in English.",
    "hi": "Answer in Hindi (Devanagari script).",
    "zh": "Answer in Simplified Chinese (Mandarin).",
}

_SYSTEM_PROMPT = (
    "You are a helpful, accurate enterprise knowledge assistant for Meridian Global Corp "
    "(a fictional 40,000-employee multinational technology company). "
    "Answer questions using ONLY the provided document excerpts. "
    "Never fabricate information. If the answer is not in the excerpts, say so clearly."
)


def get_client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise EnvironmentError("DEEPSEEK_API_KEY not set in .env")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def _build_context(chunks: list[dict]) -> tuple[str, int]:
    """Build context string from chunks, respecting MAX_CONTEXT_CHARS.

    Returns (context_str, num_chunks_used).
    """
    parts: list[str] = []
    total = 0
    for i, chunk in enumerate(chunks):
        snippet = (
            f"[{i + 1}] {chunk['source_file']} "
            f"(bu: {chunk['bu']}, lang: {chunk['lang']})\n"
            f"{chunk['text']}"
        )
        if total + len(snippet) > MAX_CONTEXT_CHARS:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts), len(parts)


def generate(query: str, chunks: list[dict], lang: str = "en") -> dict:
    """Generate a grounded answer from retrieved chunks.

    Args:
        query:  User question (PII-safe)
        chunks: Retrieved Milvus hits (list of dicts with 'text', 'source_file', etc.)
        lang:   Detected language code for the answer

    Returns:
        {
            answer:       str       — the grounded response
            sources:      list[str] — unique source filenames cited
            lang:         str       — language of the answer
            chunks_used:  int       — number of context chunks included
        }
    """
    if not chunks:
        return {
            "answer": "I could not find relevant information in the Meridian knowledge base.",
            "sources": [],
            "lang": lang,
            "chunks_used": 0,
        }

    context, n_used = _build_context(chunks)
    lang_instruction = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"])

    user_prompt = (
        f"{lang_instruction}\n\n"
        "Use the document excerpts below to answer the question. "
        "Cite sources by filename in square brackets, e.g. [hr_expense_policy.en.pdf]. "
        "If the answer is not in the excerpts, say: "
        "\"I don't have information about that in the Meridian knowledge base.\"\n\n"
        f"Document excerpts:\n{context}\n\n"
        f"Question: {query}"
    )

    client = get_client()
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=800,
    )

    answer = (resp.choices[0].message.content or "").strip()
    sources = list(dict.fromkeys(c["source_file"] for c in chunks[:n_used]))  # ordered, unique

    return {
        "answer":      answer,
        "sources":     sources,
        "lang":        lang,
        "chunks_used": n_used,
    }
