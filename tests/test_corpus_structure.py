"""
tests/test_corpus_structure.py — Sprint 1 test gate

Verifies every file in corpus/ follows the strict naming convention:
    {bu}_{slug}.{lang}.{ext}                   (PDF, MP3)
    {bu}_{slug}.{lang}_slide{NNN}.png          (slide PNG)

business_unit ∈ {hr, it_security, product, exec_comms}
lang          ∈ {en, hi, zh}
ext           ∈ {pdf, png, mp3}
"""
import re
from pathlib import Path
import pytest

CORPUS_DIR = Path(__file__).parent.parent / "corpus"

VALID_BUS = {"hr", "it_security", "product", "exec_comms"}
VALID_LANGS = {"en", "hi", "zh"}
VALID_EXTS = {".pdf", ".png", ".mp3"}

# Pattern: {bu}_{slug}.{lang}[_slide{NNN}].{ext}
FILE_PATTERN = re.compile(
    r"^(hr|it_security|product|exec_comms)"  # business unit
    r"_([a-z0-9_]+)"                          # slug
    r"\.(en|hi|zh)"                           # language
    r"(_slide\d{3})?"                         # optional slide suffix
    r"\.(pdf|png|mp3)$"                       # extension
)


def _all_corpus_files() -> list[Path]:
    if not CORPUS_DIR.exists():
        return []
    return [
        f for f in CORPUS_DIR.rglob("*")
        if f.is_file() and not f.name.startswith(".")
    ]


def test_corpus_directory_exists():
    """corpus/ directory must exist."""
    assert CORPUS_DIR.exists(), (
        f"corpus/ not found at {CORPUS_DIR}.\n"
        "Run: python scripts/generate_corpus.py"
    )


def test_corpus_not_empty():
    """corpus/ must contain at least one file."""
    files = _all_corpus_files()
    assert len(files) > 0, (
        "corpus/ exists but is empty.\n"
        "Run: python scripts/generate_corpus.py"
    )


@pytest.mark.parametrize("filepath", _all_corpus_files())
def test_filename_matches_convention(filepath: Path):
    """Every file must match the {bu}_{slug}.{lang}[_slide{NNN}].{ext} pattern."""
    name = filepath.name
    assert FILE_PATTERN.match(name), (
        f"File '{name}' does not match naming convention.\n"
        f"Expected pattern: {{bu}}_{{slug}}.{{lang}}[_slide{{NNN}}].{{ext}}\n"
        f"Valid business units: {sorted(VALID_BUS)}\n"
        f"Valid languages: {sorted(VALID_LANGS)}"
    )


@pytest.mark.parametrize("filepath", _all_corpus_files())
def test_file_in_correct_subdirectory(filepath: Path):
    """Each file must be inside a subdirectory matching its business unit."""
    name = filepath.name
    m = FILE_PATTERN.match(name)
    if not m:
        pytest.skip("filename doesn't match pattern — covered by test_filename_matches_convention")
    bu = m.group(1)
    assert filepath.parent.name == bu, (
        f"File '{name}' has business unit '{bu}' but lives in '{filepath.parent.name}/'. "
        f"Expected location: corpus/{bu}/{name}"
    )


@pytest.mark.parametrize("filepath", _all_corpus_files())
def test_file_not_empty(filepath: Path):
    """Every file must be non-empty (not a zero-byte stub)."""
    assert filepath.stat().st_size > 0, (
        f"File '{filepath.name}' is empty (0 bytes)."
    )
