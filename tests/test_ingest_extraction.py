"""
tests/test_ingest_extraction.py — Sprint 2 test gate

Unit tests for the text extraction layer.
No API calls — reads directly from corpus/ files.

PDF extraction is always tested.
Slide OCR test is skipped if pytesseract / Tesseract binary is not installed.
Audio extraction always returns "".
"""

import warnings
from pathlib import Path
import pytest

CORPUS_DIR = Path(__file__).parent.parent / "corpus"

# Pick representative fixtures
_PDF_FILES = sorted(CORPUS_DIR.glob("**/*.pdf"))[:3]  # first 3 PDFs
_SLIDE_FILES = sorted(CORPUS_DIR.glob("**/*.png"))[:2]  # first 2 slide PNGs
_AUDIO_FILES = sorted(CORPUS_DIR.glob("**/*.mp3"))[:1]  # first audio


# ── PDF ───────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(len(_PDF_FILES) == 0, reason="No PDFs in corpus/")
@pytest.mark.parametrize("pdf_path", _PDF_FILES, ids=[f.name for f in _PDF_FILES])
def test_pdf_extraction_non_empty(pdf_path: Path):
    """extract_pdf must return at least 100 characters of text."""
    from ingest.extractors import extract_pdf
    text = extract_pdf(pdf_path)
    assert len(text.strip()) >= 100, (
        f"{pdf_path.name}: extracted only {len(text.strip())} chars — "
        "expected at least 100 chars of real text."
    )


@pytest.mark.skipif(len(_PDF_FILES) == 0, reason="No PDFs in corpus/")
def test_pdf_extraction_no_exception():
    """extract_pdf must not raise for any corpus PDF."""
    from ingest.extractors import extract_pdf
    for pdf in CORPUS_DIR.rglob("*.pdf"):
        try:
            extract_pdf(pdf)
        except Exception as e:
            pytest.fail(f"extract_pdf raised for {pdf.name}: {e}")


# ── Slide PNG ─────────────────────────────────────────────────────────────────

def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@pytest.mark.skipif(len(_SLIDE_FILES) == 0, reason="No slide PNGs in corpus/")
@pytest.mark.parametrize("slide_path", _SLIDE_FILES, ids=[f.name for f in _SLIDE_FILES])
def test_slide_extraction_does_not_crash(slide_path: Path):
    """extract_slide must never raise — returns '' if OCR unavailable."""
    from ingest.extractors import extract_slide
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = extract_slide(slide_path)
    assert isinstance(result, str)  # always returns str


@pytest.mark.skipif(
    not _tesseract_available() or len(_SLIDE_FILES) == 0,
    reason="Tesseract not installed or no slides in corpus/",
)
def test_slide_extraction_non_empty_with_tesseract():
    """If Tesseract is installed, slide OCR must return some text."""
    from ingest.extractors import extract_slide
    for slide in _SLIDE_FILES:
        text = extract_slide(slide)
        assert text.strip(), f"{slide.name}: Tesseract returned empty text"


# ── Audio ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(len(_AUDIO_FILES) == 0, reason="No MP3s in corpus/")
def test_audio_extraction_deferred():
    """extract_audio must return '' (deferred to Sprint 4)."""
    from ingest.extractors import extract_audio
    result = extract_audio(_AUDIO_FILES[0])
    assert result == "", (
        "extract_audio should return '' until Sprint 4 Whisper integration. "
        f"Got: {repr(result[:80])}"
    )


# ── Dispatch ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(len(_PDF_FILES) == 0, reason="No PDFs in corpus/")
def test_extract_dispatch_pdf():
    """extract() dispatch function must handle PDFs correctly."""
    from ingest.extractors import extract
    text = extract(_PDF_FILES[0])
    assert isinstance(text, str) and text.strip()


def test_modality_inference():
    """modality_of() must classify files correctly."""
    from ingest.extractors import modality_of
    assert modality_of(Path("hr_expense_policy.en.pdf")) == "pdf"
    assert modality_of(Path("exec_comms_q3_allhands.en_slide001.png")) == "slide"
    assert modality_of(Path("exec_comms_q3_allhands.en.mp3")) == "audio"
