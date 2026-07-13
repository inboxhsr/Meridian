"""
eval/run_eval.py — Sprint 9

RAGAS evaluation runner for the Meridian pipeline.

Usage:
    # Start the FastAPI server first:
    #   uvicorn app.main:app --port 8000

    # Run full eval (live pipeline + live RAGAS scoring):
    python eval/run_eval.py

    # Dry-run on first N samples only:
    python eval/run_eval.py --limit 5

Output:
    eval/results.json — metrics JSON (not committed to git; eval artefact only)

Architecture note:
    - _call_api()       : calls POST /query on the running FastAPI server
    - _ragas_evaluate() : calls ragas.evaluate(); mocked in unit tests
    - run_eval_on_samples(): orchestrates everything; testable with mocks
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests  # type: ignore

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent  # F:\Meridian\build\
_EVAL_SET = Path(__file__).parent / "eval_set.json"
_RESULTS_OUT = Path(__file__).parent / "results.json"

# Abstention signal — matches what abstainer.py returns
_ABSTENTION_SIGNALS = [
    "insufficient grounded evidence",
    "unable to answer",
    "no relevant information",
    "could not find",
    "i don't have",
    "out of scope",
]


# ── Internal helpers (swappable for mocking in tests) ─────────────────────────

def _call_api(
    question: str,
    bu_filter: str,
    api_url: str,
    top_k: int = 5,
    timeout: int = 60,
) -> dict[str, Any]:
    """
    POST /query to the running FastAPI server and return the parsed response dict.

    Returns a dict with at minimum:
        answer (str), hits (list[dict])
    On any HTTP or network error, returns {"answer": "", "hits": []}.
    """
    payload: dict[str, Any] = {"query": question, "top_k": top_k}
    if bu_filter:
        payload["bu"] = bu_filter
    try:
        resp = requests.post(f"{api_url}/query", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Normalise hits to dicts with the keys ragas_adapter expects
        raw_hits = data.get("hits", [])
        hits = []
        for h in raw_hits:
            hits.append({
                "modality": h.get("modality", "text"),
                "chunk_text": h.get("chunk_text", ""),
                "source_file": h.get("source_file", ""),
                "doc_id": h.get("source_file", ""),
                "metadata": h.get("metadata", {}),
            })
        return {"answer": data.get("answer", ""), "hits": hits}
    except Exception as exc:  # noqa: BLE001
        logger.warning("API call failed: %s", exc)
        return {"answer": "", "hits": []}


def _ragas_evaluate(samples: list[dict]) -> dict[str, float]:
    """
    Call ragas.evaluate() on the collected samples.

    Each sample dict must have keys: question, answer, contexts, ground_truth.
    Returns a dict of metric_name → float score.

    This function is a thin wrapper kept separate so unit tests can mock it
    without importing ragas (which makes live LLM calls).
    """
    from datasets import Dataset  # type: ignore
    from ragas import evaluate  # type: ignore
    from ragas.metrics import (  # type: ignore
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_list(samples)
    result = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )
    return dict(result)


# ── Public entry point ────────────────────────────────────────────────────────

def run_eval_on_samples(
    eval_data: list[dict],
    api_url: str = "http://localhost:8000",
    sleep_between: float = 1.0,
) -> dict[str, Any]:
    """
    Run the full evaluation loop on a list of QA pairs.

    Args:
        eval_data: List of QA pair dicts (from eval_set.json).
        api_url:   Base URL of the running FastAPI server.
        sleep_between: Seconds to sleep between API calls (rate-limit courtesy).

    Returns:
        A results dict with keys: run_timestamp, n_samples, metrics, custom.
    """
    from eval.ragas_adapter import build_ragas_contexts  # type: ignore

    ragas_samples: list[dict] = []
    cross_lingual_total = 0
    cross_lingual_answered = 0
    unanswerable_total = 0
    unanswerable_abstained = 0

    for i, entry in enumerate(eval_data):
        qid = entry["id"]
        question = entry["question"]
        bu_filter = entry.get("bu_filter", "")
        ground_truth = entry["ground_truth"]
        is_cross = entry.get("source_lang") != entry.get("expected_lang")
        is_unans = bool(entry.get("unanswerable"))

        logger.info("[%d/%d] %s — %s", i + 1, len(eval_data), qid, question[:60])

        api_resp = _call_api(question, bu_filter, api_url)
        answer = api_resp.get("answer", "")
        hits = api_resp.get("hits", [])

        contexts = build_ragas_contexts(hits)
        # Ensure at least one context string so RAGAS doesn't error on empty list
        if not contexts:
            contexts = ["[No context retrieved]"]

        ragas_samples.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

        # Custom metric tallies
        if is_cross:
            cross_lingual_total += 1
            answer_lower = answer.lower()
            is_abstained = any(sig in answer_lower for sig in _ABSTENTION_SIGNALS)
            if answer and not is_abstained:
                cross_lingual_answered += 1

        if is_unans:
            unanswerable_total += 1
            answer_lower = answer.lower()
            is_abstained = any(sig in answer_lower for sig in _ABSTENTION_SIGNALS) or not answer
            if is_abstained:
                unanswerable_abstained += 1

        if sleep_between > 0:
            time.sleep(sleep_between)

    # Run RAGAS scoring
    try:
        ragas_metrics = _ragas_evaluate(ragas_samples)
    except Exception as exc:  # noqa: BLE001
        logger.error("RAGAS evaluation failed: %s", exc)
        ragas_metrics = {}

    # Custom metric computation
    cross_lingual_accuracy = (
        cross_lingual_answered / cross_lingual_total
        if cross_lingual_total > 0 else 0.0
    )
    abstention_rate = (
        unanswerable_abstained / unanswerable_total
        if unanswerable_total > 0 else 0.0
    )

    results = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(eval_data),
        "api_url": api_url,
        "metrics": {k: float(v) for k, v in ragas_metrics.items()},
        "custom": {
            "cross_lingual_accuracy": round(cross_lingual_accuracy, 4),
            "cross_lingual_total": cross_lingual_total,
            "cross_lingual_answered": cross_lingual_answered,
            "abstention_rate": round(abstention_rate, 4),
            "unanswerable_total": unanswerable_total,
            "unanswerable_abstained": unanswerable_abstained,
        },
    }
    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Meridian RAGAS Evaluation Runner")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running FastAPI server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N samples (useful for quick smoke-tests)",
    )
    parser.add_argument(
        "--output",
        default=str(_RESULTS_OUT),
        help="Path to write results JSON (default: eval/results.json)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls (default: 1.0)",
    )
    args = parser.parse_args()

    with open(_EVAL_SET, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    if args.limit:
        eval_data = eval_data[: args.limit]
        logger.info("Limiting evaluation to first %d samples.", args.limit)

    logger.info("Starting evaluation: %d samples → %s", len(eval_data), args.api_url)
    results = run_eval_on_samples(eval_data, api_url=args.api_url, sleep_between=args.sleep)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Results written to %s", out_path)
    print("\n=== Evaluation Results ===")
    print(f"  Samples evaluated  : {results['n_samples']}")
    print(f"  RAGAS metrics      : {results['metrics']}")
    print(f"  Cross-lingual acc. : {results['custom']['cross_lingual_accuracy']:.1%}")
    print(f"  Abstention rate    : {results['custom']['abstention_rate']:.1%}")


if __name__ == "__main__":
    main()
