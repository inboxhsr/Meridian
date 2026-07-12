"""
tests/test_corpus_readability.py — Sprint 1 test gate

Verifies all PDFs in the corpus contain extractable text (not blank or image-only).

Uses pdfminer.six to extract text — the same library used by the ingest pipeline.
Minimum 50 characters of extracted text per PDF to pass.
"""
from pathlib import Path
import pytest

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
MIN_CHARS = 50  # A real document should have at least this much extractable text


def _all_pdfs() -> list[Path]:
    if not CORPUS_DIR.exists():
        return []
    return list(CORPUS_DIR.rglob("*.pdf"))


@pytest.mark.parametrize("pdf_path", _all_pdfs())
def test_pdf_text_extractable(pdf_path: Path):
    """Each PDF must yield at least MIN_CHARS of extractable text via pdfminer."""
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        pytest.skip("pdfminer.six not installed — run pip install pdfminer.six")

    try:
        text = extract_text(str(pdf_path))
    except Exception as e:
        pytest.fail(
            f"pdfminer failed to open '{pdf_path.name}'.\n"
            f"  Error: {e}\n"
            f"  The file may be corrupt — try regenerating it."
        )

    char_count = len(text.strip())
    assert char_count >= MIN_CHARS, (
        f"'{pdf_path.name}' yielded only {char_count} chars of text "
        f"(need >= {MIN_CHARS}).\n"
        f"  This usually means the PDF contains only images, not real text.\n"
        f"  Regenerate with: python scripts/generate_corpus.py"
    )
