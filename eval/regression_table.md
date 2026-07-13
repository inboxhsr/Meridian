# Meridian — Regression Table
> Sprint 9 Evaluation Harness · Last Updated: 2026-07-13

This table is the single most important artifact in the project for interview purposes.
Each row represents one pipeline configuration run against the full 83-pair eval set.
Numbers are populated by running `eval/run_eval.py` against the live pipeline.

---

## Results

| Configuration | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Cross-lingual Acc. | Abstention Rate | p95 Latency |
|---|---|---|---|---|---|---|---|
| Baseline (Sprint 3 linear pipeline) | — | — | — | — | — | — | — |
| + LangGraph CRAG loop (Sprint 5) | — | — | — | — | — | — | — |
| + BGE Reranker (Sprint 6) | — | — | — | — | — | — | — |
| + Hybrid Retrieval (Sprint 7) | — | — | — | — | — | — | — |
| **Final pipeline (Sprint 7 + 8)** | — | — | — | — | — | — | — |

> **Status:** Results pending live eval run. Run `python eval/run_eval.py` with the FastAPI server running to populate.

---

## Metric Definitions

| Metric | Source | What It Measures |
|---|---|---|
| Context Precision | RAGAS standard | % of retrieved chunks that were actually relevant to the question |
| Context Recall | RAGAS standard | % of relevant chunks that were successfully retrieved |
| Faithfulness | RAGAS standard | % of answer claims supported by retrieved evidence (text-surrogate for audio/image) |
| Answer Relevancy | RAGAS standard | How directly the generated answer addresses the question |
| Cross-lingual Acc. | Custom | % of 26 cross-lingual gotcha pairs answered (non-abstention) |
| Abstention Rate | Custom | % of 5 unanswerable pairs that correctly returned no-answer |
| p95 Latency | Measured | 95th-percentile wall-clock time per query (seconds) |

---

## Resume Target Thresholds

| Metric | Target | Status |
|---|---|---|
| Faithfulness / Hallucination reduction vs. baseline | ≥ 50% reduction | Pending |
| Cross-lingual accuracy on gotcha subset | ≥ 75% | Pending |
| Abstention rate on unanswerable subset | > 0% | Pending |
| Cloud error recovery | Graceful fallback on 100% of failures | ✅ (Sprint 7) |

---

## Multimodal Groundedness Note

> Faithfulness scores for audio and image evidence are computed against **text surrogates**:
> - Audio chunks → Whisper transcript stored in `metadata.transcript` at ingest time
> - Image chunks → One-off Gemini Flash caption generated at eval time (not stored in Milvus)
>
> This is a deliberate design decision, not a gap. See `eval/ragas_adapter.py`.

---

## How to Run

```bash
# 1. Ensure Milvus is running
docker compose up milvus -d

# 2. Start the FastAPI server
uvicorn app.main:app --port 8000

# 3. Run full evaluation (83 samples)
python eval/run_eval.py

# 4. Dry-run on 5 samples only
python eval/run_eval.py --limit 5

# Output: eval/results.json
```

---

*Eval set: 83 QA pairs — 30 EN baseline, 10 ZH baseline, 5 HI baseline, 26 cross-lingual gotcha, 5 unanswerable, 7 multi-BU.*
