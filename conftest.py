"""
conftest.py — pytest root config for Meridian.

Loads .env from the build/ directory before any test runs.
All test files can rely on os.environ being populated.
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


# Load .env from the build/ directory (where this conftest.py lives)
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)
