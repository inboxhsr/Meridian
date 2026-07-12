"""
test_deepseek.py — Sprint 0 test gate

Verifies DEEPSEEK_API_KEY works against deepseek-chat.
DeepSeek is used by: Query Rewriter, Grader/Critic, Generator nodes.

DeepSeek exposes an OpenAI-compatible API at https://api.deepseek.com
so we use the openai package with a custom base_url.
"""
import os
import pytest
from openai import OpenAI


def test_deepseek_key_responds():
    """Send a minimal prompt to deepseek-chat and expect a response."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not set — add it to your .env file")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=100,  # reasoning model needs budget beyond its thinking phase
        )
    except Exception as exc:
        pytest.fail(
            f"DEEPSEEK_API_KEY failed to get a response from deepseek-chat.\n"
            f"  Error: {exc}\n"
            f"  → Check your key at https://platform.deepseek.com/api_keys"
        )

    # deepseek-v4-flash is a reasoning model: it may put output in
    # reasoning_content (thinking phase) rather than content on short prompts.
    # Accept either field as proof the API is working.
    content = response.choices[0].message.content or ""
    reasoning = getattr(response.choices[0].message, "reasoning_content", None) or ""
    assert content.strip() or reasoning.strip(), (
        f"DeepSeek returned an empty response in both content and reasoning_content.\n"
        f"  Raw response: {response}"
    )
