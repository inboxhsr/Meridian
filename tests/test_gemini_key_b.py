"""
test_gemini_key_b.py — Sprint 0 test gate

Verifies GEMINI_API_KEY_B works against gemini-2.5-flash.
KEY_B is used by the Query Rewriter and Critic nodes.
"""
import os
import pytest
import google.generativeai as genai


def test_gemini_key_b_flash_responds():
    """Send a minimal prompt to gemini-2.5-flash using KEY_B and expect a response."""
    api_key = os.environ.get("GEMINI_API_KEY_B")
    if not api_key:
        pytest.skip("GEMINI_API_KEY_B not set — run test_env.py first")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    try:
        response = model.generate_content("Reply with the single word: OK")
    except Exception as exc:
        pytest.fail(
            f"GEMINI_API_KEY_B failed to get a response from gemini-2.5-flash.\n"
            f"  Error: {exc}\n"
            f"  → Check your key is valid and has not exceeded its quota."
        )

    assert response.text is not None and response.text.strip() != "", (
        f"KEY_B got an empty response from gemini-2.5-flash.\n"
        f"  Raw response: {response}"
    )
