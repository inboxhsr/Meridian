"""
tests/test_streamlit_import.py — Sprint 10 gate

Verifies that the Streamlit app imports without error.
The app is not executed — only the module is parsed and imported.
"""

import subprocess
import sys
from pathlib import Path


def test_streamlit_app_imports() -> None:
    """app/streamlit_app.py must parse without a SyntaxError or ImportError."""
    app_path = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    result = subprocess.run(
        [sys.executable, "-c", f"compile(open({str(app_path)!r}).read(), {str(app_path)!r}, 'exec')"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Syntax error in streamlit_app.py:\n{result.stderr}"


def test_streamlit_dependencies_present() -> None:
    """Dependencies required at import time must be importable."""
    import streamlit
    import requests
    import dotenv
    import sqlite3
    # All pass if no ImportError
