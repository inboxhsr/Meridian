"""
scripts/run_ingest.py — Sprint 2: Ingest Pipeline

Reads all corpus files, extracts text, chunks, embeds (Gemini), and stores in Milvus.

Usage:
    python scripts/run_ingest.py              # ingest (skip already-ingested files)
    python scripts/run_ingest.py --reset      # drop collection, ingest everything fresh
    python scripts/run_ingest.py --dry-run    # count files/chunks without embedding

Embeddings use GEMINI_EMBEDDING_KEY.
Milvus must be running (docker-compose up -d).

Audio files are SKIPPED — transcription is deferred to Sprint 4 (Whisper).
Slide PNGs are processed only if pytesseract + Tesseract binary are installed;
otherwise they are skipped with a warning.
"""

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.extractors import extract, modality_of
from ingest.chunker import make_records
from ingest import milvus_store as store
from ingest import embedder as emb

CORPUS_DIR = Path(__file__).parent.parent / "corpus"

# Regex to parse filename: {bu}_{slug}.{lang}[_slideNNN].{ext}
_NAME_RE = re.compile(
    r"^(hr|it_security|product|exec_comms)"
    r"_([a-z0-9_]+)"
    r"\.(en|hi|zh)"
    r"(_slide\d{3})?"
    r"\.(pdf|png|mp3)$"
)


def _parse_filename(path: Path) -> dict | None:
    """Extract metadata from filename. Returns None if it doesn't match."""
    m = _NAME_RE.match(path.name)
    if not m:
        return None
    bu, slug, lang, slide_suffix, ext = m.groups()
    return {"bu": bu, "slug": slug, "lang": lang, "ext": ext}


def _corpus_files() -> list[Path]:
    """Return all corpus files (non-hidden, matching naming convention)."""
    return sorted(
        f for f in CORPUS_DIR.rglob("*")
        if f.is_file() and not f.name.startswith(".") and _parse_filename(f)
    )


def main(reset: bool = False, dry_run: bool = False) -> None:
    # ── Validate env ──────────────────────────────────────────────────────────
    if not dry_run:
        missing = [v for v in ("GEMINI_EMBEDDING_KEY", "MILVUS_HOST", "MILVUS_PORT") if not os.environ.get(v)]
        if missing:
            print(f"ERROR: Missing env vars: {missing}")
            sys.exit(1)

    files = _corpus_files()
    print(f"Corpus: {len(files)} files found in {CORPUS_DIR}")

    if dry_run:
        # Count expected chunks without calling any APIs
        total_chunks = 0
        for path in files:
            meta = _parse_filename(path)
            mod = modality_of(path)
            text = extract(path)
            if not text:
                print(f"  SKIP  [{mod.upper():5}] {path.name} (empty extraction)")
                continue
            from ingest.chunker import chunk_text
            n = len(chunk_text(text))
            print(f"  CHUNK [{mod.upper():5}] {path.name} → {n} chunks")
            total_chunks += n
        print(f"\nDRY RUN — estimated {total_chunks} chunks from {len(files)} files")
        return

    # ── Connect to Milvus ─────────────────────────────────────────────────────
    print("Connecting to Milvus …")
    mv_client = store.get_client()
    store.ensure_collection(mv_client, reset=reset)

    existing_count = store.count(mv_client)
    if existing_count > 0 and not reset:
        print(
            f"Collection '{store.COLLECTION}' already has {existing_count} records.\n"
            f"Use --reset to drop and re-ingest, or re-run to top-up with new files."
        )
        # Still continue — already_ingested() will skip duplicates

    # ── Embedding client ──────────────────────────────────────────────────────
    print("Initialising Gemini embedding client …")
    emb_client = emb.get_client()

    # ── Ingest loop ───────────────────────────────────────────────────────────
    total_inserted = 0
    total_skipped = 0
    total_errors = 0

    for path in files:
        meta = _parse_filename(path)
        mod = modality_of(path)

        # Skip already-ingested files (unless reset)
        if not reset and store.already_ingested(mv_client, path.name):
            print(f"  SKIP  [{mod.upper():5}] {path.name}")
            total_skipped += 1
            continue

        print(f"  GEN   [{mod.upper():5}] {path.name}", end=" … ", flush=True)

        try:
            # 1. Extract text
            text = extract(path)
            if not text.strip():
                print("EMPTY — skipped")
                total_skipped += 1
                continue

            # 2. Chunk
            records = make_records(
                text=text,
                source_file=path.name,
                bu=meta["bu"],
                lang=meta["lang"],
                modality=mod,
            )
            if not records:
                print("0 chunks — skipped")
                total_skipped += 1
                continue

            # 3. Embed
            def _progress(done, total):
                print(f"\r  GEN   [{mod.upper():5}] {path.name} … {done}/{total} embeddings", end="", flush=True)

            emb.embed_records(emb_client, records, on_progress=_progress)

            # 4. Store
            n = store.insert_records(mv_client, records)
            print(f"\r  OK    [{mod.upper():5}] {path.name} → {n} chunks stored    ")
            total_inserted += n

        except Exception as exc:
            print(f"\r  ERROR [{mod.upper():5}] {path.name}: {exc}    ")
            total_errors += 1

    print(
        f"\nDone — {total_inserted} chunks inserted, "
        f"{total_skipped} files skipped, "
        f"{total_errors} errors"
    )
    print(f"Collection '{store.COLLECTION}' total records: {store.count(mv_client)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meridian ingest pipeline")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the Milvus collection first")
    parser.add_argument("--dry-run", action="store_true", help="Count chunks without embedding or storing")
    args = parser.parse_args()
    main(reset=args.reset, dry_run=args.dry_run)
