# Citera: an agentic RAG system for technical support

**Author:** Abhiram ([@ABHIRAM1234](https://github.com/ABHIRAM1234))

Citera began as a group course project. Jayan Agarwal built the original
ingestion, hybrid retrieval and first agent loop. Everything the project is
about now is my work: the evaluation harness, the progressive-removal ablation,
the router and tool-calling paths, the per-request instrumentation, the API and
the deployment path, along with a substantial rewrite of the retrieval and agent
code it grew from. `git log` has the per-commit split.

---

## TL;DR

Citera is a retrieval-augmented technical-support assistant over 733 Ricoh ProcessDirector documents. Ask a natural-language question and it answers with page-level citations, or refuses when the docs do not contain the answer instead of making something up.

It started as an agentic pipeline (plan, retrieve, verify, retry) and ended up as a single retrieval call, because that is what the measurements supported. An ablation re-run at n=100 walked part of that back: the verifier still earns nothing, but the planner does help retrieval on the dev split, just not on the held-out one, so it stays behind a flag rather than deleted (see [§7](docs/EVALUATION.md#7-evaluation-and-metrics)).

**Measured** with `claude-opus-5` judging `claude-sonnet-4-6` over the full corpus, methodology and caveats in [§7](docs/EVALUATION.md#7-evaluation-and-metrics):

> 0.96 groundedness `[0.95-0.98]`, 0.97 correctness `[0.94-0.99]`, 1.00 citation precision, 0.98 answer-vs-refuse, 0.94 retrieval recall@5
>
> $0.0157 and 10.1s per query, on 1 LLM call

Measured on **100 questions** with a 70/30 dev/holdout split. An earlier 10-question set gave flattering numbers with intervals twice as wide. Growing the benchmark moved groundedness 0.98 -> 0.96 and revealed a behaviour-match failure the small set could not see.

A three-config progressive-removal ablation compares the full Planner -> Retriever -> Verifier -> retry loop against progressively simpler pipelines. On the dev split (70 questions, judged):

| Config | Pipeline | Calls | Cost/query | Latency | Evidence recall | Grounded | Correct |
|---|---|---|---|---|---|---|---|
| A | retrieve -> synthesize | 1.0 | $0.0156 | 10.2s | 0.94 | 0.96 | 0.97 |
| B | + planner | 2.0 | $0.0262 | 14.3s | 1.00 | 0.98 | 0.99 |
| C | + verifier/retry | 3.0 | $0.0414 | 16.5s | 1.00 | 0.97 | 0.99 |

The verifier (C) adds no evidence recall over B and is slightly worse on the judged metrics, so it stays off. The planner (B) takes evidence recall from 0.94 to 1.00 on dev, but on the held-out 30 questions A and B score an identical 0.93, so the benefit does not replicate. Config A is the default: the planner's edge is real on one split, gone on the other, and not worth 1.7x the cost per query on every question. It stays behind `USE_PLANNER` for corpora where retrieval is weaker. Full per-split numbers and the earlier n=10 run in [§7](docs/EVALUATION.md#7-evaluation-and-metrics).

Later work held up under harder tests: a judged ablation on a hand-built **multi-hop** set (the case the 100-question set was missing) still puts the planner's gain inside the judge noise floor; the **holdout** ablation, once judged, confirmed the dev-only pattern; a **cross-provider bakeoff** found `gemini-3.6-flash` holds answer quality at 1/50th the cost per query while `gpt-4o-mini` drops correctness 0.21; **multi-turn** follow-up handling (history-aware query rewriting) answers follow-ups about as well as cold questions; and a **QLoRA fine-tune** of Llama 3.1 8B, self-hosted on one GPU, distilled the synthesizer's skill closely enough to land correctness inside Sonnet's noise floor, though groundedness stays a real gap ([`finetune/`](finetune/)). All in [§7](docs/EVALUATION.md#7-evaluation-and-metrics).

I also added an optional Azure AI Search backend for the same chunks. Its semantic ranker matches the local cross-encoder on R@5 and does better on R@1, and it is much faster. See [the Azure section](docs/EVALUATION.md#azure-ai-search-as-a-second-backend).

Brackets are 95% percentile-bootstrap confidence intervals.

**Stack:** Python · LangGraph · Claude · ChromaDB (dense) + BM25 + Reciprocal Rank Fusion · optional Azure AI Search · Streamlit · pytest + GitHub Actions CI · Docker.

```bash
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-your-key" > .env   # then add PDFs to data/
streamlit run app/main.py
```

---

## Observability

Every request is traced with per-stage spans, token counts, cost, and which chunks produced the answer. Traces are plain JSONL files in `traces/`, and LangSmith is an optional extra. Look at them with `python -m src.trace_view`. Each answer also has a thumbs up or down, stored by trace id, and thumbs-down questions can be exported as eval candidates to label. More in [docs/DESIGN.md](docs/DESIGN.md#observability).

---

## 1. Problem statement

Build a technical-support assistant that answers complex, multi-part questions about RICOH ProcessDirector using **only** the supplied documentation, and that refuses rather than guesses when the documentation does not contain the answer.

Field technicians and support engineers lose significant time searching documentation for specific procedures, error-code resolutions, and configuration steps. The system needs to ingest that documentation, understand natural-language questions, retrieve the relevant passages, and generate accurate, cited answers with zero hallucination.

The thing that actually shapes the design is the corpus: 733 individual help-topic articles, most of them a single page, not a handful of long manuals (median chunk: 307 words; 1,314 chunks total). So the core retrieval task is **selecting the right document out of 733**, not locating a passage within a long document. Almost every design decision below follows from that, see [§5](docs/DESIGN.md#5-data-handling-and-preprocessing).

**End user:** Ricoh field service technicians, help desk agents, and customers seeking self-service support.

**Why is this important?** Faster resolution times reduce operational costs, improve customer satisfaction, and let technicians focus on complex problems instead of manual searching.

---

## 2. Why this problem

- **Real-world impact:** Technical support is a multi-billion dollar industry; AI-assisted retrieval meaningfully cuts the time engineers spend searching documentation.
- **Technical depth:** the problem spans ingestion, hybrid retrieval, grounded generation, and strict hallucination control, not just a chatbot wrapper.
- **A question worth answering:** does an agentic retry loop actually help here? Building one and then measuring it away turned out to be the most instructive part of the project.
- **Measurable evaluation:** a fixed question set makes retrieval and generation quality something you can regress against rather than argue about.

---

## 3. Solution overview

**Citera** is an agentic AI technical support system that:

1. **Ingests** Ricoh PDF manuals using PyMuPDF with metadata-preserving chunking (500 words, 50-word overlap).
2. **Retrieves** relevant passages via a **hybrid engine** combining semantic vector search (ChromaDB + MiniLM) and keyword search (BM25), fused with Reciprocal Rank Fusion.
3. **Runs** a LangGraph state machine whose composition is set by measurement: the Planner and Verifier stages exist but are **off by default**, because an ablation showed the verifier earns nothing and the planner's help does not replicate across splits ([§7](docs/EVALUATION.md#7-evaluation-and-metrics)).
4. **Generates** grounded answers with strict `[Document Name, Page X]` citations - refusing to answer when evidence is insufficient.
5. **Visualises** the full reasoning process in a "Glass Box" Streamlit dashboard.
6. **Polyglot Support:** Automatically detects user language (e.g., Spanish, Japanese, Hindi) and answers in that language while preserving English citations.

The shape of the pipeline came out of the ablation rather than being assumed up front, and two stages ended up switched off as a result.

---

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
- **Agentic loop, built, measured, and switched off by default.** The rationale was that single-pass retrieval misses evidence on multi-part questions and a plan-then-verify loop catches the gaps. At n=100 the verifier still earns nothing, and the planner helps retrieval on the dev split but not the held-out one, so the loop is not worth 1.7x the cost per query here. It stays behind config flags for a corpus where retrieval is weaker ([§7](docs/EVALUATION.md#7-evaluation-and-metrics)).

---

## 5. Data handling and preprocessing

The corpus is 733 Ricoh ProcessDirector help-topic PDFs (about 223 MB, in `data/`, not committed). Most are a single page, so the hard part is picking the right document out of 733, not finding a passage in a long manual. PyMuPDF extracts the text page by page and keeps `source_document` and `page_number`. Text is chunked at 500 words with 50 words of overlap, which gives 1,322 chunks, stored in ChromaDB (vectors) and a pickled BM25 index. 31 pages with screenshots or diagrams also get a vision-model description added before chunking. The full write-up is in [docs/DESIGN.md](docs/DESIGN.md#5-data-handling-and-preprocessing).

---

## 6. Modeling and AI strategy

The default path is one retrieval call followed by one Claude Sonnet call at temperature 0, which lowers variance but does not make the output deterministic. The synthesizer has to cite `[Document Name, Page X]` for every claim and say "Information unavailable in provided documents" when the evidence is not there. The planner and verifier prompts exist but are off by default. Retrieval is ChromaDB (MiniLM) plus BM25, merged with RRF (k=60), top 5. OpenAI and Gemini can be swapped in through `src/llm_factory.py`. More in [docs/DESIGN.md](docs/DESIGN.md#6-modeling-and-ai-strategy).

---

## 7. Evaluation and metrics

The headline numbers are in the TL;DR above. The full write-up is in [docs/EVALUATION.md](docs/EVALUATION.md#7-evaluation-and-metrics). It covers the 100-question benchmark and how the judge is set up, the ablation (n=10, n=100, holdout, multi-hop), tool calling and routing, multi-turn follow-ups, the embedding and reranker sweeps, query rewriting (HyDE and multi-query), the Azure AI Search comparison, the cross-provider bakeoff, the fine-tuned model, the prompt-injection guardrail test, and corrections I made to my own earlier results.

---

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

- ~~Table-heavy content may have reduced retrieval accuracy due to PDF text extraction limitations.~~ There turned out to be no real tables in this corpus, only two-column property-reference lists, which text extraction already handles fine. The actual gap was screenshots and workflow diagrams: 31 pages across 28 documents now get a vision-model description appended to their extracted text ([§5](docs/DESIGN.md#5-data-handling-and-preprocessing)).

---

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

---

## 10. How to run the project

### Prerequisites
- Python 3.11+ (developed on 3.13)
- Anthropic API key

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/abhirammv2000/Ricoh.git
cd Ricoh

# 2. Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env

# 5. Place Ricoh PDFs in data/
# (Copy all provided PDF manuals into the data/ directory)
```

### Run the Application
```bash
# Launch the Streamlit dashboard
streamlit run app/main.py
# Or equivalently:
python -m streamlit run app/main.py
```

### Run the Evaluation
```bash
# Quality harness: groundedness, correctness, recall@k, citation precision
python -m src.eval_harness            # full run, uses the LLM judge (needs API key)
python -m src.eval_harness --no-judge # objective metrics only, no API calls

# Legacy latency/citation smoke test
python -m src.evaluate                # outputs evaluation_results.csv + evaluation_report.md
```
Results and methodology are in [§7 Evaluation & Metrics](docs/EVALUATION.md#7-evaluation-and-metrics).

### Run Individual Components
```bash
python -m src.ingest       # PDF ingestion only
python -m src.retriever    # Retrieval smoke test
python -m src.agent        # Agent smoke test
```

### Live public demo (Ngrok)

To share a live demo link:

```bash
# 1. Install Ngrok (https://ngrok.com/download)
# Or via Chocolatey on Windows:
choco install ngrok

# 2. Run Streamlit locally
streamlit run app/main.py

# 3. In a separate terminal, expose port 8501
ngrok http 8501

# 4. Share the generated https://xxxx.ngrok-free.app link
```

---

## 11. Testing and CI

```bash
pip install -r requirements-dev.txt
pytest                      # offline unit tests (LLM is mocked, no API key needed)
```

The suite covers the logic most likely to break silently:
- **Chunking invariants**, size cap, overlap preservation, page-provenance isolation, deterministic IDs (`tests/test_ingest.py`)
- **RRF fusion math**, rank merging, score accumulation, `final_k` truncation (`tests/test_retriever.py`)
- **Agent control flow**, retry routing, planner JSON parsing (incl. fenced/malformed output), verifier verdict normalisation (`tests/test_agent.py`)
- **Eval metrics**, citation extraction, recall@k, citation precision, refusal detection (`tests/test_eval_metrics.py`)

[GitHub Actions](.github/workflows/ci.yml) runs `pytest` on every push/PR (Python 3.11). No secrets required. The tests mock the LLM.

A second CI job is a **retrieval regression gate** (`python -m eval.ci_gate`): it runs real hybrid retrieval against the committed `demo_index/` on the seed questions and fails the build if recall@1/3/5 drops below `eval/ci_baseline.json`. The unit suite never touches a real index, so this is what would catch an RRF or fusion bug that still passes every mocked test. The same job also checks that `demo_index` has not drifted from the benchmark it serves: if `eval/ground_truth.json` gains a question whose expected document is not baked into the index, the build fails with a "rebuild demo_index" message. `demo_index` is a small curated slice, so the recall check is a smoke gate, not a quality measurement; the full-corpus numbers come from the paid harness in [§7](docs/EVALUATION.md#7-evaluation-and-metrics).

## 12. Optional: cross-encoder reranker

A query-aware cross-encoder reranker (off by default to keep the base torch-free) can be enabled for higher precision@k:

```bash
pip install -r requirements-reranker.txt
RERANKER_ENABLED=true streamlit run app/main.py
```

It fuses a larger candidate pool, re-scores with `cross-encoder/ms-marco-MiniLM-L-6-v2`, and trims to the top-k. Lazy-loaded and cached, so the default lightweight path is untouched. Whether it earns its latency on this corpus is measured by `python -m eval.sweep_embeddings --measure --rerank` ([§7](docs/EVALUATION.md#7-evaluation-and-metrics)).

## 13. Deployment

Beyond the local Ngrok demo, the project ships a reproducible container path. See **[DEPLOYMENT.md](DEPLOYMENT.md)** for Docker, Render (one-click `render.yaml`), and Streamlit Cloud instructions.

```bash
docker build -t citera .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... -v "$PWD/data:/app/data" citera
```

## 14. Project status: demo vs production

Being straight about what this is:

**Built and working:** hybrid retrieval + RRF, agentic verify-retry loop, grounded/cited generation with refusal, multi-lingual answers, Glass Box UI with a per-request cost/attribution drill-down, a quality eval harness, a judged multi-turn eval and a judged multi-hop ablation, a cross-provider (Anthropic / OpenAI / Gemini) abstraction with a bakeoff, a unit-test suite + CI with a retrieval regression gate, per-request tracing (local JSONL plus opt-in LangSmith), an optional semantic answer cache, SDK-level retry/timeout handling, a rate limit and optional password on the public demo, and a containerised deploy path.

**Deliberately out of scope (next steps for true production):** real auth (not just a shared demo password), a secrets manager, sampling production traffic back into the eval set, human-labelled judge calibration (worksheet prepared), and a CI-built full-corpus index instead of the baked demo subset. These are tracked in [DEPLOYMENT.md](DEPLOYMENT.md).

## 15. Repository structure

`app/` has the Streamlit dashboard, `src/` the pipeline (ingest, retriever, agent, tools, router, instrumentation), `eval/` the benchmarks and sweeps, `finetune/` the QLoRA fine-tune, and `tests/` the offline pytest suite. The full tree is in [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md).

---

## 16. Roadmap

The roadmap table, with what was done and what is left, is in [docs/ROADMAP_STATUS.md](docs/ROADMAP_STATUS.md).
