# RicohLibrary - Implementation Walkthrough

## Overview

RicohLibrary is an agentic AI technical support system built in 6 phases over a single development sprint. It ingests Ricoh PDF manuals, indexes them with a hybrid retrieval engine, and answers questions using a LangGraph state machine with verify-and-retry logic.

**Stack:** Python 3.11 · PyMuPDF · ChromaDB · BM25 · LangGraph · Claude Sonnet · Streamlit

---

## Phase 1: PDF Ingestion Pipeline

**File:** `src/ingest.py`

- Uses **PyMuPDF** (`fitz`) to extract text page-by-page from all PDFs in `data/`
- Chunks text using a **sliding window**: ~500 words per chunk, 50-word overlap
- Preserves critical metadata on every chunk:
  - `source_document` - original PDF filename
  - `page_number` - page the text came from
  - `chunk_id` - unique identifier for de-duplication
- Returns a flat list of chunk dicts ready for indexing

**Design decision:** Word-based chunking (not character-based) to preserve semantic coherence. Overlap ensures no answer is split across chunk boundaries.

---

## Phase 2: Hybrid Retrieval Engine

**File:** `src/retriever.py`

Two search backends combined with rank fusion:

### Semantic Search (ChromaDB)
- Uses `all-MiniLM-L6-v2` sentence transformer (local, no API key)
- Vectors stored in `chroma_db/` - persists across restarts
- Finds semantically similar passages even with different wording

### Keyword Search (BM25)
- `rank_bm25.BM25Okapi` over tokenised chunk text
- Serialised to disk via `pickle` (`bm25_index.pkl` + `bm25_chunks.pkl`)
- Catches exact matches that vector search misses (error codes like `SC542`, model numbers like `IM C3500`)

### Reciprocal Rank Fusion (RRF)
- Merges ranked results from both backends using `RRF(k=60)`
- Rank-based fusion - avoids the problem of incomparable score scales between cosine similarity and BM25 scores
- Returns top-5 fused results per query

**Design decision:** Hybrid > pure vector because technical documentation is full of exact identifiers that semantic search alone misses.

---

## Phase 3: LangGraph Agentic State Machine

**File:** `src/agent.py`

### State Definition (`AgentState`)
```python
class AgentState(TypedDict):
    user_query: str
    sub_queries: list[str]
    entities: list[str]          # error codes, model numbers
    retrieved_evidence: list[dict]
    verification_status: str     # "SUFFICIENT" | "INSUFFICIENT"
    final_answer: str
    iterations: int              # max 2
```

### 4 Graph Nodes

| Node | Role | Key Detail |
|---|---|---|
| **Planner** | Decomposes query → sub-queries + entities | JSON output parsed with regex fallback |
| **Retriever** | Two-pass hybrid search | Pass 1: sub-queries, Pass 2: entity-boosted |
| **Verifier** | Binary SUFFICIENT/INSUFFICIENT verdict | Defaults to SUFFICIENT on parse failure |
| **Synthesizer** | Generates cited answer | Strict `[Document, Page X]` format |

### Graph Topology
```
START → Planner → Retriever → Verifier ─┬─→ Synthesizer → END
           ↑                             │
           └──── (INSUFFICIENT & <2) ────┘
```

### Retry Logic
- On INSUFFICIENT: Planner receives list of already-searched sources, generates broader sub-queries
- Hard cap at 2 iterations to prevent infinite loops / API cost runaway
- If still insufficient after 2 iterations, Synthesizer runs anyway and states uncertainty

### Prompt Engineering
- **Planner prompt:** Extracts entities + decomposes. Includes retry context on 2nd pass.
- **Verifier prompt:** Single-word response enforced. Defaults to SUFFICIENT on ambiguous output.
- **Synthesizer prompt:** Rules: cite every claim, refuse to guess, use `[Document Name, Page X]` format.

---

## Phase 4: Evaluation Pipeline

**File:** `src/evaluate.py`

- Runs all 10 official hackathon questions through the agent
- Captures per-question: answer text, latency, source documents cited
- Outputs:
  - `evaluation_results.csv` - machine-readable results
  - `evaluation_report.md` - judge-friendly formatted report
- **Results:** 10/10 questions answered, avg 13.9s latency, hallucination control verified

---

## Phase 5: Streamlit Glass Box UI

**File:** `app/main.py`

### Features
- **Chat interface** with message history and session state
- **Glass Box expander** under each answer showing:
  - 🧠 Sub-queries the Planner generated
  - 🏷️ Extracted entities (error codes, model numbers)
  - 📚 Evidence cards with document name + page number + text snippet
  - ⏱️ Latency, 🔄 iteration count, ✅ verification status
- **Sidebar:** LLM provider/model, index size, query count, reset button

### Technical Details
- `sys.path` manipulation to resolve `src` package from `app/` directory
- `@st.cache_resource` for retriever to avoid re-loading index on reruns
- `run_agent_full()` returns complete `AgentState` (not just the answer) for Glass Box
- Custom CSS with dark theme, gradient header, Inter font, evidence card styling

---

## Phase 6: README & Documentation

**File:** `README.md`

Follows the official HackVerse 2026 template with all 12 required sections:
1. Problem Statement → 2. Why We Chose This → 3. Solution Overview → 4. Architecture → 5. Data Handling → 6. Modeling Strategy → 7. Evaluation → 8. Business Impact → 9. Tech Stack → 10. How to Run → 11. Repo Structure → 12. Rubric Alignment → Compliance Statement

---

## Key Bug Fixes

### BM25 Persistence Bug
**Problem:** BM25 index lived only in memory - empty on restart.
**Fix:** Serialize BM25 index + chunk data to pickle files after `build_index()`, auto-load in `__init__()`.

### ChromaDB Telemetry Spam
**Problem:** ChromaDB telemetry errors flooding terminal output.
**Fix:** `os.environ["ANONYMIZED_TELEMETRY"] = "False"` at the very top of `config.py` before any imports. Suppressed noisy loggers (`chromadb`, `httpx`, `httpcore`) via centralized logging config.

### Dependency Conflict
**Problem:** `langchain-core==0.3.34` conflicted with `langchain-anthropic>=0.3.53`.
**Fix:** Loosened to `langchain-core>=0.3.34`.

---

## Configuration

**File:** `src/config.py`

All project constants centralized:
- Paths: `DATA_DIR`, `CHROMA_DIR`, `BM25_INDEX_PATH`, `BM25_CHUNKS_PATH`
- Retrieval: `RETRIEVAL_TOP_K=10`, `RETRIEVAL_FINAL_K=5`, `CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`
- LLM: `DEFAULT_LLM_PROVIDER="anthropic"`
- Logging: Centralized `logging.basicConfig` + library-level suppression

**File:** `src/llm_factory.py`

Provider-agnostic factory: `get_llm()` returns a LangChain chat model based on `LLM_PROVIDER` env var. Currently uses Claude Sonnet (`claude-sonnet-4-20250514`), swappable to OpenAI/Google with one config change.
