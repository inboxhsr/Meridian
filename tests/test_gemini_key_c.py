"""
test_gemini_key_c.py — Sprint 0 test gate

Verifies GEMINI_API_KEY_C works against gemini-2.5-pro.
KEY_C is used by the Generator node — the only node using the Pro model.
"""
import os
import pytest
import google.generativeai as genai


def test_gemini_key_c_pro_responds():
    """Send a minimal prompt to gemini-2.5-pro using KEY_C and expect a response."""
    api_key = os.environ.get("GEMINI_API_KEY_C")
    if not api_key:
        pytest.skip("GEMINI_API_KEY_C not set — run test_env.py first")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-pro")

    try:
        response = model.generate_content("Reply with the single word: OK")
    except Exception as exc:
        pytest.fail(
            f"GEMINI_API_KEY_C failed to get a response from gemini-2.5-pro.\n"
            f"  Error: {exc}\n"
            f"  → Check your key is valid. Note: gemini-2.5-pro may have stricter "
            f"rate limits on free-tier accounts."
        )

    assert response.text is not None and response.text.strip() != "", (
        f"KEY_C got an empty response from gemini-2.5-pro.\n"
        f"  Raw response: {response}"
    )
