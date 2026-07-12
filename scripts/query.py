"""
scripts/query.py — Sprint 3: RAG Query Pipeline CLI

Runs a question through the full Meridian RAG pipeline:
  Route → Retrieve → Generate

Usage:
    python scripts/query.py "What is the travel expense reimbursement policy?"
    python scripts/query.py "IT security incident procedure" --bu it_security
    python scripts/query.py "差旅报销上限是多少？" --top-k 3
    python scripts/query.py "query here" --skip-pii   # skip PII guard (faster)

Requirements: Milvus running (docker-compose up -d), .env populated.
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.router    import route
from pipeline.retriever import retrieve
from pipeline.generator import generate

BU_CHOICES = ["", "hr", "it_security", "product", "exec_comms"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Meridian RAG query pipeline")
    parser.add_argument("query",    help="Question to ask the knowledge base")
    parser.add_argument("--bu",     default="", choices=BU_CHOICES,
                        metavar="BU", help="Scope to a business unit: hr | it_security | product | exec_comms")
    parser.add_argument("--top-k",  type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--skip-pii", action="store_true",
                        help="Skip PII guard (saves 1 Gemini call; use for testing)")
    args = parser.parse_args()

    bar = "=" * 60

    print(f"\n{bar}")
    print(f"  Query : {args.query}")
    if args.bu:
        print(f"  BU    : {args.bu}")
    print(bar)

    # ── Step 1: Route ──────────────────────────────────────────────────────────
    print("\n[1/3] Routing …")
    routing = route(args.query, bu_filter=args.bu, skip_pii=args.skip_pii)
    print(f"  Language : {routing['lang']}")
    print(f"  BU filter: {routing['bu_filter'] or '(all)'}")
    if routing["pii_flagged"]:
        print(f"  ⚠  PII detected — query was redacted before retrieval")
        print(f"  Safe query: {routing['safe_query']}")

    # ── Step 2: Retrieve ───────────────────────────────────────────────────────
    print(f"\n[2/3] Retrieving top-{args.top_k} chunks …")
    chunks = retrieve(
        routing["safe_query"],
        top_k=args.top_k,
        bu_filter=routing["bu_filter"],
    )

    if not chunks:
        print("  No relevant chunks found.")
        print("  Suggestions: broaden your query, remove the --bu filter, or check that")
        print("  the ingest pipeline has been run (scripts/run_ingest.py).")
        sys.exit(0)

    for i, c in enumerate(chunks):
        print(f"  [{i+1}] {c['source_file']}  score={c['score']:.4f}  lang={c['lang']}")

    # ── Step 3: Generate ───────────────────────────────────────────────────────
    print("\n[3/3] Generating grounded answer …")
    result = generate(routing["safe_query"], chunks, lang=routing["lang"])

    print(f"\n{bar}")
    print("  ANSWER")
    print(bar)
    print(result["answer"])

    print(f"\n  Sources ({len(result['sources'])} document(s)):")
    for src in result["sources"]:
        print(f"    • {src}")

    print(f"  Chunks used: {result['chunks_used']} / {len(chunks)}")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()
