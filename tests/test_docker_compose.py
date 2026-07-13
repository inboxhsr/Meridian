"""
tests/test_docker_compose.py — Sprint 10 gate

Verifies that docker-compose.yml is valid (docker compose config succeeds).
"""

import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).parent.parent


def _docker_available() -> bool:
    """Return True if docker CLI is reachable."""
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def test_docker_compose_config_valid() -> None:
    """docker compose config must validate without error."""
    if not _docker_available():
        import pytest
        pytest.skip("Docker not available — skipping compose validation test")

    result = subprocess.run(
        ["docker", "compose", "config"],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        timeout=30,
    )
    assert result.returncode == 0, (
        f"docker compose config failed (exit {result.returncode}):\n{result.stderr}"
    )


def test_docker_compose_has_app_service() -> None:
    """docker compose config output must include the 'app' service."""
    if not _docker_available():
        import pytest
        pytest.skip("Docker not available — skipping compose validation test")

    result = subprocess.run(
        ["docker", "compose", "config"],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        timeout=30,
    )
    assert "app:" in result.stdout, "docker-compose.yml must define an 'app' service"
