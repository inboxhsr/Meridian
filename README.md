# Meridian — Build Directory

This directory contains the full implementation of the **Meridian** cross-lingual multimodal agentic RAG system.

> For architecture decisions, tech stack justifications, and corpus design, see [`../Planning/project_charter.md`](../Planning/project_charter.md).
> For the sprint plan and tech-stack modifications, see [`../Planning/implementation_plan.md`](../Planning/implementation_plan.md).
> For current sprint status, see [`../Planning/sprint_tracker.md`](../Planning/sprint_tracker.md).

---

## What Is Meridian?

A cross-lingual, multimodal agentic RAG system for a fictional 40,000-employee multinational.

- Ask a question in **Hindi** → retrieve the answer from a **Chinese PDF** → respond in Hindi with a citation.
- Three languages (English, Hindi, Chinese), three modalities (text, image, audio), one shared embedding space.
- A LangGraph agentic pipeline with a CRAG-style self-correction loop, driven entirely by the **Gemini API**.

---

## Directory Layout

```
build/
├── corpus/                    # Raw source files (PDFs, PNGs, MP3s)
│   ├── hr/
│   ├── it_security/
│   ├── product/
│   └── exec_comms/
│
├── ingest/                    # Ingestion pipeline
│   ├── ingest.py              # Entry point: python ingest.py --source ./corpus
│   ├── chunker.py             # Language-aware sentence-packed chunking
│   ├── embedder.py            # Gemini Embedding 2 calls (text / image / audio)
│   ├── transcribe.py          # Whisper-base transcription (audio only)
│   ├── schema.py              # Milvus collection schema definition
│   └── milvus_client.py       # Connection + collection lifecycle
│
├── pipeline/                  # LangGraph agent
│   ├── graph.py               # Graph definition + compile
│   ├── state.py               # LangGraph state schema
│   ├── nodes/
│   │   ├── pii_redactor.py
│   │   ├── router.py
│   │   ├── query_rewriter.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   ├── critic.py
│   │   ├── generator.py
│   │   └── abstainer.py
│   └── prompts/               # System prompts per node (versioned)
│
├── eval/                      # Evaluation harness
│   ├── eval_set.json          # ~80 hand-written QA pairs
│   ├── run_eval.py            # RAGAS evaluation runner
│   ├── ragas_adapter.py       # Text-surrogate logic for multimodal groundedness
│   └── regression_table.md    # Before/after results per pipeline config
│
├── observability/             # Observability layer
│   ├── schema.sql             # SQLite table definition
│   └── dashboard.py           # Streamlit observability dashboard
│
├── app/                       # Streamlit demo UI
│   └── app.py
│
├── tests/                     # pytest test suite (mirrors build structure)
│   ├── conftest.py
│   ├── test_env.py
│   ├── test_milvus_connection.py
│   └── ...                    # One test file per sprint gate
│
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── requirements.txt
└── README.md                  # This file
```

---

## Quickstart

### Prerequisites

- Python 3.11
- Docker Desktop (for Milvus Standalone)
- 3 Google AI Studio API keys (see `.env.example`)

### Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env and add your 3 Gemini API keys

# 4. Start Milvus Standalone
docker compose up milvus -d

# 5. Verify environment
pytest tests/test_env.py tests/test_milvus_connection.py -v
```

### Ingest corpus

```bash
python ingest/ingest.py --source ./corpus
```

### Run the demo

```bash
streamlit run app/app.py
```

### Run evaluation

```bash
python eval/run_eval.py
```

### Run all tests

```bash
pytest tests/ -v
```

---

## Model Configuration

| Node | Model | API Key |
|---|---|---|
| PII Redactor | Gemini 2.5 Flash | `GEMINI_API_KEY_A` |
| Router | Gemini 2.5 Flash | `GEMINI_API_KEY_A` |
| Query Rewriter | Gemini 2.5 Flash | `GEMINI_API_KEY_B` |
| Grader / Critic | Gemini 2.5 Flash | `GEMINI_API_KEY_B` |
| Generator | Gemini 2.5 Pro | `GEMINI_API_KEY_C` |
| Embedder | Gemini Embedding 2 | `GEMINI_EMBEDDING_KEY` |
| Reranker | BGE-reranker-v2-m3 (local) | — |
| Transcription | Whisper-base (local) | — |

---

## Build Progress

See [`../Planning/sprint_tracker.md`](../Planning/sprint_tracker.md) for live sprint status.

---

*Part of the Meridian portfolio project. See `../Planning/` for full architecture and documentation.*
