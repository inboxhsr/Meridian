#!/usr/bin/env python3
"""
scripts/run_e2e.py — Meridian E2E Integration Test Runner

Convenience wrapper around pytest for the full-stack integration tests.
Checks pre-conditions before running so pytest output is clean.

Usage:
    python scripts/run_e2e.py                    # full suite
    python scripts/run_e2e.py --quick            # skip slow latency test
    python scripts/run_e2e.py --api http://...   # custom API URL
    python scripts/run_e2e.py --check-only       # only verify pre-conditions

Pre-conditions checked:
    1. FastAPI server reachable at API_URL/health
    2. Milvus connected (milvus: true in /health)
    3. Corpus loaded (corpus_chunks >= 100 in /health)
    4. GEMINI_API_KEY_A, GEMINI_EMBEDDING_KEY, DEEPSEEK_API_KEY in environment
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import requests  # type: ignore
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

_ROOT = Path(__file__).parent.parent  # F:\Meridian\build\


# ── Pre-condition checks ───────────────────────────────────────────────────────

def check_api(api_url: str) -> tuple[bool, str]:
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        if r.status_code != 200:
            return False, f"GET /health returned HTTP {r.status_code}"
        data = r.json()
        if not data.get("milvus"):
            return False, "Milvus not reachable — start Docker: docker compose up milvus -d"
        chunks = data.get("corpus_chunks", 0)
        if chunks < 100:
            return False, (
                f"corpus_chunks={chunks} (expected ≥100). "
                "Run: python scripts/run_ingest.py"
            )
        return True, f"API OK — milvus=true, corpus_chunks={chunks}"
    except requests.exceptions.ConnectionError:
        return False, (
            f"Cannot reach {api_url}. "
            "Start the server: uvicorn app.main:app --port 8000"
        )
    except Exception as e:
        return False, f"Unexpected error: {e}"


def check_env() -> tuple[bool, str]:
    required = ["GEMINI_API_KEY_A", "GEMINI_EMBEDDING_KEY", "DEEPSEEK_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        return False, f"Missing environment variables: {', '.join(missing)}"
    return True, "All API keys present"


def print_check(label: str, ok: bool, detail: str) -> None:
    icon = "✅" if ok else "❌"
    print(f"  {icon}  {label}: {detail}")


def run_prechecks(api_url: str) -> bool:
    print("\n=== Meridian E2E Pre-condition Check ===\n")
    api_ok, api_msg = check_api(api_url)
    env_ok, env_msg = check_env()

    print_check("FastAPI + Milvus + Corpus", api_ok, api_msg)
    print_check("API Keys (.env)          ", env_ok, env_msg)

    all_ok = api_ok and env_ok
    if all_ok:
        print("\n  All pre-conditions met. Running E2E tests...\n")
    else:
        print("\n  ❌ Pre-conditions NOT met. Fix the issues above, then re-run.\n")
        print("  Quick-start commands:")
        print("    docker compose up milvus -d")
        print("    python scripts/run_ingest.py")
        print("    uvicorn app.main:app --port 8000")
        print("    python scripts/run_e2e.py\n")
    return all_ok


# ── pytest runner ──────────────────────────────────────────────────────────────

def run_tests(api_url: str, quick: bool, extra_args: list[str]) -> int:
    """Run pytest for test_e2e_integration.py. Returns pytest exit code."""
    env = os.environ.copy()
    env["MERIDIAN_API_URL"] = api_url  # picked up by the test module

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_e2e_integration.py",
        "-v",
        "--tb=short",
    ]

    if quick:
        cmd += ["-m", "not slow"]
        print("  [--quick] Skipping @pytest.mark.slow tests (latency guard)\n")

    cmd += extra_args

    result = subprocess.run(cmd, cwd=str(_ROOT), env=env)
    return result.returncode


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meridian E2E Integration Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api",
        default=os.environ.get("MERIDIAN_API_URL", "http://localhost:8000"),
        metavar="URL",
        help="FastAPI server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip slow tests (latency regression guard)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify pre-conditions; do not run tests",
    )

    args, extra = parser.parse_known_args()

    ok = run_prechecks(args.api)

    if args.check_only:
        sys.exit(0 if ok else 1)

    if not ok:
        sys.exit(1)

    code = run_tests(args.api, args.quick, extra)
    sys.exit(code)


if __name__ == "__main__":
    main()
