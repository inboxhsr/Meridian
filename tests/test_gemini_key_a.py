"""
test_gemini_key_a.py — Sprint 0 test gate

Verifies GEMINI_API_KEY_A works against gemini-2.5-flash.
KEY_A is used by the PII Redactor and Router nodes.

SDK: google-genai (replaces deprecated google-generativeai, EOL Nov 2025)
"""
import os
import pytest
from google import genai


def test_gemini_key_a_flash_responds():
    """Send a minimal prompt to gemini-2.5-flash using KEY_A and expect a response."""
    api_key = os.environ.get("GEMINI_API_KEY_A")
    if not api_key:
        pytest.skip("GEMINI_API_KEY_A not set — run test_env.py first")

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Reply with the single word: OK",
        )
    except Exception as exc:
        pytest.fail(
            f"GEMINI_API_KEY_A failed to get a response from gemini-2.5-flash.\n"
            f"  Error: {exc}\n"
            f"  → Check your key is valid at https://aistudio.google.com"
        )

    assert response.text is not None and response.text.strip() != "", (
        f"KEY_A got an empty response from gemini-2.5-flash.\n"
        f"  Raw response: {response}"
    )
