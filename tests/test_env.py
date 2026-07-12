"""
test_env.py — Sprint 0 test gate

Verifies all required environment variables are present and non-empty.
Fails immediately with a clear message if any key is missing.
"""
import os
import pytest


REQUIRED_VARS = [
    "GEMINI_API_KEY_A",
    "GEMINI_API_KEY_B",
    "GEMINI_API_KEY_C",
    "GEMINI_EMBEDDING_KEY",
    "MILVUS_HOST",
    "MILVUS_PORT",
]


@pytest.mark.parametrize("var_name", REQUIRED_VARS)
def test_env_var_present_and_non_empty(var_name):
    """Each required env var must exist and not be an empty string."""
    value = os.environ.get(var_name)
    assert value is not None, (
        f"Missing env var: {var_name}\n"
        f"  → Copy .env.example to .env and fill in your values."
    )
    assert value.strip() != "", (
        f"Env var {var_name} is set but empty.\n"
        f"  → Open .env and add your actual value."
    )
