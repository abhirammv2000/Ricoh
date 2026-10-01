# Overview

## 1. Problem statement

Build a technical-support assistant that answers complex, multi-part questions about RICOH ProcessDirector using **only** the supplied documentation, and that refuses rather than guesses when the documentation does not contain the answer.

Field technicians and support engineers lose significant time searching documentation for specific procedures, error-code resolutions, and configuration steps. The system needs to ingest that documentation, understand natural-language questions, retrieve the relevant passages, and generate accurate, cited answers with zero hallucination.

The thing that actually shapes the design is the corpus: 733 individual help-topic articles, most of them a single page, not a handful of long manuals (median chunk: 307 words; 1,314 chunks total). So the core retrieval task is **selecting the right document out of 733**, not locating a passage within a long document. Almost every design decision below follows from that, see [§5](DESIGN.md#5-data-handling-and-preprocessing).

**End user:** Ricoh field service technicians, help desk agents, and customers seeking self-service support.

**Why is this important?** Faster resolution times reduce operational costs, improve customer satisfaction, and let technicians focus on complex problems instead of manual searching.


## 2. Why this problem

- **Real-world impact:** Technical support is a multi-billion dollar industry; AI-assisted retrieval meaningfully cuts the time engineers spend searching documentation.
- **Technical depth:** the problem spans ingestion, hybrid retrieval, grounded generation, and strict hallucination control, not just a chatbot wrapper.
- **A question worth answering:** does an agentic retry loop actually help here? Building one and then measuring it away turned out to be the most instructive part of the project.
- **Measurable evaluation:** a fixed question set makes retrieval and generation quality something you can regress against rather than argue about.


## 3. Solution overview

**Citera** is an agentic AI technical support system that:

1. **Ingests** Ricoh PDF manuals using PyMuPDF with metadata-preserving chunking (500 words, 50-word overlap).
2. **Retrieves** relevant passages via a **hybrid engine** combining semantic vector search (ChromaDB + MiniLM) and keyword search (BM25), fused with Reciprocal Rank Fusion.
3. **Runs** a LangGraph state machine whose composition is set by measurement: the Planner and Verifier stages exist but are **off by default**, because an ablation showed the verifier earns nothing and the planner's help does not replicate across splits ([§7](EVALUATION.md#7-evaluation-and-metrics)).
4. **Generates** grounded answers with strict `[Document Name, Page X]` citations - refusing to answer when evidence is insufficient.
5. **Visualises** the full reasoning process in a "Glass Box" Streamlit dashboard.
6. **Polyglot Support:** Automatically detects user language (e.g., Spanish, Japanese, Hindi) and answers in that language while preserving English citations.

The shape of the pipeline came out of the ablation rather than being assumed up front, and two stages ended up switched off as a result.


## 4. Architecture and system design

```
User Question
    |
    v
CONDENSE (only on a follow-up: rewrite it to stand on its own; §7)
    |
    v
LangGraph state machine

  default (what the ablation settled on):
      RETRIEVER  ->  SYNTHESIZER

  optional, off unless USE_PLANNER / USE_VERIFIER are set:
      PLANNER  ->  RETRIEVER  ->  VERIFIER  ->  SYNTHESIZER
                      ^              |
                      |              |  pass 1: sub-queries
                      |              |  pass 2: entity-boosted
                      +--------------+
                        while INSUFFICIENT and iter < 2
    |
    v
Cited answer plus the Glass Box view
```

### Pipeline Components

| Component | Technology | Purpose |
|---|---|---|
| PDF Ingestion | PyMuPDF | Extract text + preserve page metadata |
| Chunking | Custom sliding window | 500-word chunks, 50-word overlap |
| Semantic Search | ChromaDB + all-MiniLM-L6-v2 | Dense vector similarity (offline) |
| Keyword Search | BM25Okapi | Exact match for error codes & model numbers |
| Fusion | Reciprocal Rank Fusion (k=60) | Rank-based merging (scale-invariant) |
| Reasoning | LangGraph StateGraph | Explicit, auditable control flow; planner/verifier stages configurable and off by default |
| LLM | Claude Sonnet 4.6 (agent) / Opus 5 (eval judge) | Grounded generation; low temperature to reduce variance |
| UI | Streamlit | Glass Box dashboard with chat interface |

### Why this architecture?
- **Hybrid retrieval** because pure vector search misses exact matches on error codes (`SC542`) and model numbers (`IM C3500`), while pure BM25 misses semantic similarity.
- **RRF fusion** because BM25 and cosine similarity scores are incommensurable - rank-based fusion avoids score normalisation issues.
- **Agentic loop, built, measured, and switched off by default.** The rationale was that single-pass retrieval misses evidence on multi-part questions and a plan-then-verify loop catches the gaps. At n=100 the verifier still earns nothing, and the planner helps retrieval on the dev split but not the held-out one, so the loop is not worth 1.7x the cost per query here. It stays behind config flags for a corpus where retrieval is weaker ([§7](EVALUATION.md#7-evaluation-and-metrics)).

## 8. Business impact and actionability

### How this helps decision-makers
- **Support engineers:** Get instant, cited answers instead of manually searching hundreds of separate documentation articles, faster time-to-answer.
- **Help desk managers:** Glass Box transparency lets supervisors verify answer quality before sending to customers.
- **Training:** New technicians can learn by exploring the agent's reasoning process.

### Real-world usability
- Runs entirely offline (except LLM API) - deployable in air-gapped environments with a local LLM.
- Modular architecture allows swapping LLM providers (Anthropic/OpenAI/Google) via a single config change.

### Limitations
- Requires pre-ingested PDF manuals; no real-time document updates.
- LLM API latency (~10-15s) may be too slow for live phone support - could be improved with smaller/local models.

Fixed, not just documented:

- ~~Table-heavy content may have reduced retrieval accuracy due to PDF text extraction limitations.~~ There turned out to be no real tables in this corpus, only two-column property-reference lists, which text extraction already handles fine. The actual gap was screenshots and workflow diagrams: 31 pages across 28 documents now get a vision-model description appended to their extracted text ([§5](DESIGN.md#5-data-handling-and-preprocessing)).


## 9. Tech stack

| Category | Technology |
|---|---|
| Language | Python 3.11+ (developed on 3.13) |
| PDF Parsing | PyMuPDF 1.25.3 |
| Vector Database | ChromaDB 0.6.3 (local, all-MiniLM-L6-v2) |
| Keyword Search | rank_bm25 0.2.2 |
| Optional search backend | Azure AI Search (`requirements-azure.txt`) |
| Agentic Framework | LangGraph 0.2.74 |
| LLM | Claude Sonnet via langchain-anthropic 0.3.12 |
| UI | Streamlit 1.42.0 |
| Configuration | python-dotenv 1.0.1 |
