"""
ingest/extractors.py — Sprint 2

Text extraction from each corpus modality.

| Modality | Library        | Notes                                    |
|----------|----------------|------------------------------------------|
| PDF      | pdfminer.six   | always available                         |
| PNG      | pytesseract    | optional — falls back to "" if Tesseract |
|          |                | binary not installed                     |
| MP3      | (deferred)     | Whisper transcription planned Sprint 4   |
"""

import re
import warnings
from pathlib import Path


# ── PDF ───────────────────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> str:
    """Extract text from a PDF using pdfminer. Always available."""
    from pdfminer.high_level import extract_text
    text = extract_text(str(path))
    return _clean(text)


# ── Slide PNG ─────────────────────────────────────────────────────────────────

def extract_slide(path: Path) -> str:
    """Extract text from a slide PNG using pytesseract OCR.

    Returns "" if pytesseract or the Tesseract binary is not installed.
    Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
    """
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(str(path)).convert("RGB")
        text = pytesseract.image_to_string(img, config="--psm 6")
        return _clean(text)
    except ImportError:
        warnings.warn(
            "pytesseract not installed — slide OCR skipped. "
            "Install: pip install pytesseract + Tesseract binary"
        )
        return ""
    except Exception as exc:
        warnings.warn(f"Slide OCR failed for {path.name}: {exc}")
        return ""


# ── Audio MP3 ─────────────────────────────────────────────────────────────────

_whisper_model = None   # module-level cache — loaded once per process


def _get_whisper_model(size: str = "tiny"):
    global _whisper_model
    if _whisper_model is None:
        import whisper as _whisper
        _whisper_model = _whisper.load_model(size)
    return _whisper_model


def extract_audio(path: Path) -> str:
    """Transcribe MP3 using openai-whisper (tiny model, CPU-only, ~39 MB).

    First call downloads the model from OpenAI's CDN.
    Requires ffmpeg: https://ffmpeg.org/download.html (winget install ffmpeg)

    Falls back to "" if openai-whisper or ffmpeg is not installed.
    """
    try:
        model = _get_whisper_model("tiny")
        result = model.transcribe(str(path), fp16=False)   # fp16=False → CPU safe
        return _clean(result.get("text", ""))
    except ImportError:
        warnings.warn(
            "openai-whisper not installed — audio transcription skipped. "
            "Install: pip install openai-whisper"
        )
        return ""
    except Exception as exc:
        warnings.warn(f"Whisper transcription failed for {path.name}: {exc}")
        return ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Collapse excessive whitespace and strip."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def modality_of(path: Path) -> str:
    """Infer modality from file extension and naming convention."""
    name = path.name
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith(".mp3"):
        return "audio"
    if name.endswith(".png") and "_slide" in name:
        return "slide"
    return "unknown"


def extract(path: Path) -> str:
    """Dispatch to the correct extractor based on modality."""
    m = modality_of(path)
    if m == "pdf":
        return extract_pdf(path)
    if m == "slide":
        return extract_slide(path)
    if m == "audio":
        return extract_audio(path)
    warnings.warn(f"Unknown modality for {path.name} — skipping")
    return ""
