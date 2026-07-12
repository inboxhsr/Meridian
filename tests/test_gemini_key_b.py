"""
test_gemini_key_b.py — Sprint 0 test gate

Verifies GEMINI_API_KEY_B works against gemini-2.5-flash.
KEY_B is used by the Query Rewriter and Critic nodes.

SDK: google-genai (replaces deprecated google-generativeai, EOL Nov 2025)
"""
import os
import pytest
from google import genai


def test_gemini_key_b_flash_responds():
    """Send a minimal prompt to gemini-2.5-flash using KEY_B and expect a response."""
    api_key = os.environ.get("GEMINI_API_KEY_B")
    if not api_key:
        pytest.skip("GEMINI_API_KEY_B not set — run test_env.py first")

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Reply with the single word: OK",
        )
    except Exception as exc:
        pytest.fail(
            f"GEMINI_API_KEY_B failed to get a response from gemini-2.5-flash.\n"
            f"  Error: {exc}\n"
            f"  → Check your key is valid at https://aistudio.google.com"
        )

    assert response.text is not None and response.text.strip() != "", (
        f"KEY_B got an empty response from gemini-2.5-flash.\n"
        f"  Raw response: {response}"
    )
