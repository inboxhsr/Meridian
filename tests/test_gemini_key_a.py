"""
test_gemini_key_a.py — Sprint 0 test gate

Verifies GEMINI_API_KEY_A works against gemini-2.5-flash.
KEY_A is used by the PII Redactor and Router nodes.
"""
import os
import pytest
import google.generativeai as genai


def test_gemini_key_a_flash_responds():
    """Send a minimal prompt to gemini-2.5-flash using KEY_A and expect a response."""
    api_key = os.environ.get("GEMINI_API_KEY_A")
    if not api_key:
        pytest.skip("GEMINI_API_KEY_A not set — run test_env.py first")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    try:
        response = model.generate_content("Reply with the single word: OK")
    except Exception as exc:
        pytest.fail(
            f"GEMINI_API_KEY_A failed to get a response from gemini-2.5-flash.\n"
            f"  Error: {exc}\n"
            f"  → Check your key is valid and has not exceeded its quota."
        )

    assert response.text is not None and response.text.strip() != "", (
        f"KEY_A got an empty response from gemini-2.5-flash.\n"
        f"  Raw response: {response}"
    )
