"""
tests/test_corpus_coverage.py — Sprint 1 test gate

Verifies the corpus has sufficient coverage across:
  - Minimum total file counts
  - All 3 languages present (en, hi, zh)
  - All 4 business units present
  - All 3 modalities present (pdf, png/slides, mp3)
"""
from pathlib import Path
from collections import defaultdict
import re
import pytest

CORPUS_DIR = Path(__file__).parent.parent / "corpus"

FILE_PATTERN = re.compile(
    r"^(hr|it_security|product|exec_comms)"
    r"_([a-z0-9_]+)"
    r"\.(en|hi|zh)"
    r"(_slide\d{3})?"
    r"\.(pdf|png|mp3)$"
)

# Minimum thresholds — Sprint 1 targets ~36 source docs
MIN_PDFS = 15          # 5 HR×3 lang + 2 IT_sec×3 lang + 5 product×2 lang
MIN_SLIDE_PNGS = 20    # 8 decks × 5 slides (we test a lower bound)
MIN_AUDIOS = 3
MIN_BUS = 4            # hr, it_security, product, exec_comms
MIN_LANGS = 3          # en, hi, zh


def _scan():
    """Return a dict with aggregated counts from corpus/."""
    stats = {
        "pdfs": 0, "pngs": 0, "mp3s": 0,
        "langs": set(), "bus": set(), "slide_decks": set(),
    }
    if not CORPUS_DIR.exists():
        return stats
    for f in CORPUS_DIR.rglob("*"):
        if not f.is_file():
            continue
        m = FILE_PATTERN.match(f.name)
        if not m:
            continue
        bu, slug, lang, slide_suffix, ext = m.groups()
        stats["langs"].add(lang)
        stats["bus"].add(bu)
        if ext == "pdf":
            stats["pdfs"] += 1
        elif ext == "png":
            stats["pngs"] += 1
            stats["slide_decks"].add((bu, slug, lang))
        elif ext == "mp3":
            stats["mp3s"] += 1
    return stats


@pytest.fixture(scope="module")
def corpus_stats():
    return _scan()


def test_minimum_pdf_count(corpus_stats):
    """At least MIN_PDFS PDF documents must exist."""
    count = corpus_stats["pdfs"]
    assert count >= MIN_PDFS, (
        f"Found {count} PDFs, need at least {MIN_PDFS}.\n"
        "Run: python scripts/generate_corpus.py"
    )


def test_minimum_slide_png_count(corpus_stats):
    """At least MIN_SLIDE_PNGS slide PNGs must exist."""
    count = corpus_stats["pngs"]
    assert count >= MIN_SLIDE_PNGS, (
        f"Found {count} slide PNGs, need at least {MIN_SLIDE_PNGS}.\n"
        "Run: python scripts/generate_corpus.py"
    )


def test_minimum_audio_count(corpus_stats):
    """At least MIN_AUDIOS MP3 audio files must exist."""
    count = corpus_stats["mp3s"]
    assert count >= MIN_AUDIOS, (
        f"Found {count} MP3 files, need at least {MIN_AUDIOS}.\n"
        "Run: python scripts/generate_corpus.py"
    )


def test_all_languages_present(corpus_stats):
    """All 3 languages (en, hi, zh) must be present in the corpus."""
    missing = {"en", "hi", "zh"} - corpus_stats["langs"]
    assert not missing, (
        f"Missing language(s) in corpus: {sorted(missing)}.\n"
        "Run: python scripts/generate_corpus.py"
    )


def test_all_business_units_present(corpus_stats):
    """All 4 business units must be present in the corpus."""
    required = {"hr", "it_security", "product", "exec_comms"}
    missing = required - corpus_stats["bus"]
    assert not missing, (
        f"Missing business unit(s) in corpus: {sorted(missing)}.\n"
        "Run: python scripts/generate_corpus.py"
    )


def test_all_modalities_present(corpus_stats):
    """All 3 modalities (PDF, PNG/slides, MP3) must be present."""
    missing = []
    if corpus_stats["pdfs"] == 0:
        missing.append("PDF")
    if corpus_stats["pngs"] == 0:
        missing.append("PNG slides")
    if corpus_stats["mp3s"] == 0:
        missing.append("MP3 audio")
    assert not missing, (
        f"Missing modality/modalities: {missing}.\n"
        "Run: python scripts/generate_corpus.py"
    )


def test_multilingual_pdfs_present(corpus_stats):
    """Hindi (hi) and Chinese (zh) PDFs must exist — not just English."""
    # Check by rescanning specifically for hi and zh PDFs
    hi_pdfs = zh_pdfs = 0
    if CORPUS_DIR.exists():
        for f in CORPUS_DIR.rglob("*.hi.pdf"):
            hi_pdfs += 1
        for f in CORPUS_DIR.rglob("*.zh.pdf"):
            zh_pdfs += 1
    assert hi_pdfs >= 2, f"Found {hi_pdfs} Hindi PDFs, need at least 2."
    assert zh_pdfs >= 2, f"Found {zh_pdfs} Chinese PDFs, need at least 2."
