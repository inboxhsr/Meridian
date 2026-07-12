"""
test_gemini_key_c.py — Sprint 0 test gate

Verifies GEMINI_API_KEY_C works against gemini-2.5-flash.
KEY_C is used by the Generator node.

NOTE: gemini-2.5-pro is PAYWALLED on free-tier AI Studio keys (as of April 2026).
      All Meridian nodes use gemini-2.5-flash across all three keys.
      The 3-key rotation still provides rate-limit headroom per the original design.

SDK: google-genai (replaces deprecated google-generativeai, EOL Nov 2025)
"""
import os
import pytest
from google import genai


def test_gemini_key_c_flash_responds():
    """Send a minimal prompt to gemini-2.5-flash using KEY_C and expect a response."""
    api_key = os.environ.get("GEMINI_API_KEY_C")
    if not api_key:
        pytest.skip("GEMINI_API_KEY_C not set — run test_env.py first")

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Reply with the single word: OK",
        )
    except Exception as exc:
        pytest.fail(
            f"GEMINI_API_KEY_C failed to get a response from gemini-2.5-flash.\n"
            f"  Error: {exc}\n"
            f"  → Check your key is valid at https://aistudio.google.com\n"
            f"  → gemini-2.5-pro is paywalled on free-tier keys; flash is used instead."
        )

    assert response.text is not None and response.text.strip() != "", (
        f"KEY_C got an empty response from gemini-2.5-flash.\n"
        f"  Raw response: {response}"
    )
