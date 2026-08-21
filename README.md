# Citera: an agentic RAG system for technical support

**Author:** Abhiram ([@ABHIRAM1234](https://github.com/ABHIRAM1234))

---

## TL;DR

**Citera** is a retrieval-augmented technical-support assistant over **733 Ricoh ProcessDirector documents**. Ask a natural-language question; it answers with page-level citations and **refuses when the docs don't contain the answer** instead of hallucinating.

It was built as an agentic pipeline (plan → retrieve → verify → retry) and then **measured back down to a single retrieval call**, because that is what the evidence supported.

**Measured** with `claude-opus-5` judging `claude-sonnet-4-6` over the full corpus, methodology and caveats in [§7](#7-evaluation-and-metrics), full decision log in [ROADMAP.md](ROADMAP.md):

> **0.96** groundedness `[0.95-0.98]` · **0.97** correctness `[0.94-0.99]` · **1.00** citation precision · **0.98** answer-vs-refuse · **0.94** retrieval recall@5
>
> **$0.0157** and **10.1s** per query, on **1 LLM call**

Measured on **100 questions** with a 70/30 dev/holdout split. An earlier 10-question set gave flattering numbers with intervals twice as wide. Growing the benchmark moved groundedness 0.98 → 0.96 and revealed a behaviour-match failure the small set could not see.

**The headline result: every stage had to earn its cost against an ablation, and two of them failed.** Switching the planner and verifier off made the system cheaper and faster and no worse on any quality metric.

A three-config progressive-removal ablation compared the full Planner → Retriever → Verifier → retry loop against progressively simpler pipelines:

| Config | Pipeline | Calls | Cost/query | Latency | Evidence recall | Grounded | Correct |
|---|---|---|---|---|---|---|---|
| **A** | retrieve → synthesize | **1.0** | **$0.0159** | **9.9s** | **1.00** | 0.98 | **1.00** |
| B | + planner | 2.0 | $0.0207 | 13.9s | 0.88 | 0.99 | 0.97 |
| C | + verifier/retry *(old default)* | 3.4 | $0.0490 | 15.6s | 0.88 | 0.98 | 0.98 |

Config A matched or beat the full pipeline on **every quality metric** at **3.1× lower cost** and **1.6× lower latency**. Not one question was measurably better under C. Config A is now the default; the stages remain behind config flags rather than deleted, because the ablation's scope is this corpus and this question set, see [§7](#7-evaluation-and-metrics).

Brackets are 95% percentile-bootstrap confidence intervals. The headline figures above are n=100. The ablation table is n=10, and its intervals are correspondingly wide. Numbers are reported with the uncertainty the sample size supports rather than as bare point estimates.

**Stack:** Python · LangGraph · Claude · ChromaDB (dense) + BM25 + Reciprocal Rank Fusion · Streamlit · pytest + GitHub Actions CI · Docker.

```bash
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-your-key" > .env   # then add PDFs to data/
streamlit run app/main.py
```

---

## Observability

Every request is traced: a trace id, per-stage spans (retrieval and LLM), token
counts, derived cost, and **chunk attribution**, for any stored answer you can
ask which chunks produced it, at what rank and RRF score.

```bash
python -m src.trace_view              # recent traces + total recorded spend
python -m src.trace_view --last       # full span breakdown of one request
python -m src.trace_view --slowest 3  # worst latencies
python -m src.trace_view --doc aiw00a13.pdf   # every request that used a document
```

Traces are append-only JSONL in `traces/`, no service required to run this
repo. For a hosted timeline on top of that, LangSmith tracing is wired in and
opt-in: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` and every agent run
streams to LangSmith as well, with the local JSONL still written. It stays off
by default so reproduction never depends on an external account.

An optional semantic answer cache (also off by default) short-circuits the
synthesizer for a repeat or near-duplicate question. A hit is recorded as a
zero-cost span, so the cost dashboard shows the saving rather than hiding it, and
the threshold is set from measured cosine similarity rather than guessed.

**Honest gaps:** the app ships an aggregate cost and latency dashboard
(`src/perf.py`), but per-request drill-down, the chunk attribution behind one
past trace, is CLI-only. Production traffic is not sampled back into the eval
set.

---

## 1. Problem statement

Build a technical-support assistant that answers complex, multi-part questions about RICOH ProcessDirector using **only** the supplied documentation, and that refuses rather than guesses when the documentation does not contain the answer.

Field technicians and support engineers lose significant time searching documentation for specific procedures, error-code resolutions, and configuration steps. The system needs to ingest that documentation, understand natural-language questions, retrieve the relevant passages, and generate accurate, cited answers with zero hallucination.

**The constraint that actually shapes the design:** the corpus is **733 individual help-topic articles, most of them a single page**, not a handful of long manuals (median chunk: 307 words; 1,314 chunks total). So the core retrieval task is **selecting the right document out of 733**, not locating a passage within a long document. Almost every design decision below follows from that, see [§5](#5-data-handling-and-preprocessing).

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
3. **Runs** a LangGraph state machine whose composition is set by measurement: the Planner and Verifier stages exist but are **off by default**, because an ablation showed the single-retrieval path matched or beat the full loop at 3.1× lower cost ([§7](#7-evaluation-and-metrics)).
4. **Generates** grounded answers with strict `[Document Name, Page X]` citations - refusing to answer when evidence is insufficient.
5. **Visualises** the full reasoning process in a "Glass Box" Streamlit dashboard.
6. **Polyglot Support:** Automatically detects user language (e.g., Spanish, Japanese, Hindi) and answers in that language while preserving English citations.

**What makes it distinctive:** the pipeline's composition is an evidence-backed decision rather than an assumption. Every stage had to prove it earned its cost, and two of them failed that test and were switched off.

---

## 4. Architecture and system design

```
User Question
     ↓
┌──────────────────────────────────────────────────────────┐
│               LangGraph State Machine                     │
│                                                           │
│  DEFAULT (measured):                                      │
│     RETRIEVER ──→ SYNTHESIZER                       │
│                                                           │
│  OPTIONAL (USE_PLANNER / USE_VERIFIER - off by default):  │
│   PLANNER ──→ RETRIEVER (2-pass) ──→ VERIFIER  │
│      ↑            │ Pass 1: sub-queries      │           │
│      │            │ Pass 2: entity-boosted   │           │
│      └──── (INSUFFICIENT & iter < 2) ────────┘           │
│                                  ↓                        │
│                          SYNTHESIZER                   │
└──────────────────────────────────────────────────────────┘
     ↓
Cited Answer + Glass Box Visualisation
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
- **Agentic loop, built, measured, and switched off.** The original rationale was that single-pass retrieval misses evidence on multi-part questions and the verify-retry pattern catches the gaps. That was a hypothesis, and on this corpus it turned out to be false: the loop retrieved *worse* than the raw question and cost 3.1× more. The stages remain behind config flags because the rationale may well hold on a corpus with weaker retrieval. It simply does not hold here ([§7](#7-evaluation-and-metrics)).

---

## 5. Data handling and preprocessing

- **Dataset:** Official Ricoh ProcessDirector (RPD) documentation, **733 PDFs (~223 MB)**, stored in `data/` (gitignored due to size). Honest note: these are **individual help-topic articles**, most of which are a *single page* each, rather than a handful of 100+ page manuals. This is why nearly every citation reads "Page 1", and it means the real retrieval challenge here is **picking the right document out of 733**, not pinpointing a page within a long manual. The page-level citation machinery still works (and would matter for true multi-page manuals), but we call out the corpus shape rather than overstate it.
- **Extraction:** PyMuPDF extracts raw text page-by-page, preserving `source_document` and `page_number` metadata throughout.
- **Chunking strategy:** Sliding window of ~500 words with 50-word overlap. Word-based (not character-based) to keep semantic coherence. Overlap ensures no answer is lost at chunk boundaries.
- **Tokenisation (BM25):** Simple lowercase whitespace split - intentionally basic because error codes like `SC542` don't benefit from stemming.
- **Storage:** ChromaDB (vector index) + pickled BM25 (keyword index), both persisted to `chroma_db/` for fast restarts.
- **Limitations:** Table-heavy PDF pages may lose structure during text extraction. Future work could add table parsing.

---

## 6. Modeling and AI strategy

### LLM: Claude Sonnet (Anthropic)
- **Why:** Strong instruction-following, reliable JSON output for the planner, low hallucination rate, and cheap enough to run 4-5 calls per question.
- **On temperature:** set to 0.0 to *reduce* output variance. It does **not** make generation deterministic. Temperature 0 has never guaranteed identical outputs. Measured run-to-run variation in the planner's sub-queries is the main source of end-to-end variance in this system; retrieval itself is bit-identical across runs.
- **Alternative considered:** GPT-4o (OpenAI) - switched to Claude due to API availability constraints.

### Prompt Engineering (4 specialised prompts)
1. **Planner prompt:** Decomposes queries into sub-queries + extracts entities. Outputs structured JSON. Includes retry-aware context injection.
2. **Verifier prompt:** Binary SUFFICIENT/INSUFFICIENT verdict. Defaults to SUFFICIENT on ambiguous output to prevent infinite loops.
3. **Synthesizer prompt:** Strict citation rules - every claim must cite `[Document Name, Page X]`. Refuses to answer when evidence is missing.
4. **Retry context:** On INSUFFICIENT verdict, the Planner receives a list of already-searched sources to broaden the next search.

### Retrieval Strategy
- **Semantic:** ChromaDB with local all-MiniLM-L6-v2 embeddings (no API key needed).
- **Keyword:** BM25Okapi over full chunk corpus.
- **Fusion:** RRF(k=60) merges rank positions, returning top-5 fused results per sub-query.

### Hallucination Control
- The Synthesizer is instructed to say *"Information unavailable in provided documents"* when evidence is insufficient - validated in our evaluation (see Section 7).

---

## 7. Evaluation and metrics

### Test set: 10 seed questions, expanded to 100
1. What property do I set if I want the printers to enable after a restart?
2. How much RAM does the primary server need for document-level processing?
3. How much hard drive space should I allocate for DB2 logs?
4. Does RPD work with FusionPro?
5. What operating system does RPD run on?
6. How do I create a workflow?
7. What programs does RPD integrate with?
8. What is the command to shut down RPD?
9. How do I use locations?
10. What inserters does RPD support?

These 10 were the original question set and seeded the benchmark. The eval set is now **100 questions** (`eval/generate_questions.py`, 70/30 dev/holdout), and the headline metrics are measured on all 100. The 10 are kept here because the ablation in this section was run on them.

_(Per-question latency and scores: [eval/eval_report_n100.md](eval/eval_report_n100.md) for the current run, [eval/eval_report.md](eval/eval_report.md) for the original 10.)_

### Two levels of evaluation

**1. Latency / citation smoke test** (`src/evaluate.py`), *legacy*. Runs the 10 questions and records latency and whether a citation regex matched. It never checked whether answers were *correct* or *faithful*, which is why the harness below exists.

**2. Quality harness** (`src/eval_harness.py` → `eval/metrics.json`, `eval/eval_report.md`), **the one to look at.**

| Metric | LLM-judged? | What it actually measures, and what it does *not* |
|---|---|---|
| **Evidence recall** | No | Did the expected document reach the synthesizer **across all passes and retries**? Deliberately *not* called recall@k: the accumulated evidence ranges from 5 to 40+ chunks, so calling it "@k" (k=5) would flatter the retriever. |
| **Retriever recall@1/5/20** | No | The retriever **in isolation**, raw question, one pass, no planner. Comparing against evidence recall separates a *retrieval* failure from a *planning* failure. |
| **Citation precision** | No | Are cited documents present in the evidence? Catches **fabricated filenames only**, it does *not* verify that the cited document supports the claim. Real attribution correctness needs claim→span checking (not yet built). |
| **Groundedness** | Yes | Is every claim supported by the retrieved evidence? |
| **Answer correctness** | Yes | Does the answer convey the expected key facts, or refuse correctly? |
| **Behaviour match** | No | Did it answer when it should and refuse only when it should? |

```bash
python -m src.eval_harness            # full quality run (uses the LLM judge)
python -m src.eval_harness --no-judge # objective metrics only, no API calls
python -m eval.verify_unanswerable    # audit the "refuse" labels against the corpus
python -m eval.judge_variance         # measure the judge's own noise floor
```

### Methodology: why the judge is a different model

The judge runs on **`claude-opus-5`** while the agent runs on **`claude-sonnet-4-6`** (`JUDGE_MODEL` in `src/config.py`). This is not incidental: an earlier version of this harness used the **same model as both agent and judge**, which is the weakest possible configuration: a model grading its own output exhibits self-preference bias, and published work on LLM judges finds raw agreement badly overstates chance-corrected agreement.

Switching to an independent, stronger judge moved groundedness 0.977 → 0.98 and correctness 0.97 → 0.98.

**That difference means nothing, and it is worth being precise about why.** `python -m eval.judge_variance` scores an identical answer against identical evidence five times and reports the spread:

| Case | Groundedness across 5 identical runs | Spread |
|---|---|---|
| Q8 (unambiguous) | 1.00, 1.00, 1.00, 1.00, 1.00 | **0.00** |
| Q9 (borderline) | 0.60, 0.55, 0.55, 0.60, 0.55 | **0.05** |
| Q7 (borderline) | 0.65, 0.70, 0.70, 0.60, 0.70 | **0.10** |

The judge is perfectly stable on clear-cut answers and jitters by up to 0.10 on borderline ones, and borderline answers are precisely the ones that move an aggregate. A 0.003 shift in the mean is **more than an order of magnitude below the instrument's own noise floor**, before even accounting for agent nondeterminism producing different answers between runs.

So the honest conclusion is: **this eval cannot detect whether self-preference was inflating the scores.** The judge was changed because grading your own output is methodologically indefensible regardless of what the number does, not because the change was shown to matter. An earlier draft of this README claimed the scores "barely moved, so self-preference was not materially inflating them". That was a confounded comparison (different ground truth *and* different agent outputs between the two runs) asserting a conclusion the data cannot support.

**The general rule this establishes:** no claim about a metric change on this benchmark is credible until the change exceeds the judge's measured noise floor. That is a large part of why expanding the eval set is priority 1.

### Measured results

Full corpus (733 PDFs → 1,314 chunks), agent `claude-sonnet-4-6`, judge `claude-opus-5`, **100 questions** with a 70/30 dev/holdout split. Generated by `src/eval_harness.py` → [eval/eval_report_n100.md](eval/eval_report_n100.md) / [eval/metrics_n100.json](eval/metrics_n100.json). Brackets are 95% percentile-bootstrap CIs:

| Metric | Mean | 95% CI | n |
|---|---|---|---|
| **Behaviour-match rate** | **0.98** | n/a | 100 |
| **Groundedness** | **0.96** | [0.95, 0.98] | 100 |
| **Answer correctness** | **0.97** | [0.94, 0.99] | 100 |
| **Citation precision** | **1.00** | [1.00, 1.00] | 100 |
| **Evidence recall** | **0.94** | [0.89, 0.98] | 100 |
| **Mean latency** | **10.1s** | max 24.5s | 100 |

The earlier 10-question run reported 1.00 behaviour match, 0.98 groundedness and 0.98 correctness at 18.8s. Those numbers are **superseded, not deleted**: they are what a small sample looks like when it flatters you. Growing the set moved groundedness down 0.02, cut the intervals roughly in half, and surfaced the behaviour-match failure that 10 questions could not see. The old run is preserved in [eval/eval_report.md](eval/eval_report.md) / [eval/metrics.json](eval/metrics.json).

**The retriever in isolation.** This is the most actionable result in the project:

| Depth | Recall |
|---|---|
| recall@1 | 0.78 |
| recall@3 | 0.89 |
| **recall@5** | **0.94** |

At **production settings** (`top_k=10, final_k=5`), retrieval on the raw question reaches 0.94 recall@5 across all 100 questions, so the retriever is not the main bottleneck. The pipeline wrapped around it was:

| Path (n=10 ablation) | Recall | LLM calls |
|---|---|---|
| Raw question, single retrieval | **1.00** (8/8) | 0 |
| Full agentic pipeline | **0.88** | ~4 |

The planner degrades Q1 and Q9 by rewriting the question into sub-queries that retrieve worse than the original, and never improves any question. **The most defensible next change to this system is to delete work, not add it.** Route simple lookups straight to retrieval and reserve the agentic path for questions that demonstrably need it.

### The ablation: does each stage earn its cost?

Retrieval recall is not answer quality, so "the planner hurts retrieval" was not sufficient grounds to remove it. The agent gathers *more* evidence on some questions (Q4: 17 chunks, Q7: 8), and extra context could plausibly raise groundedness even while document recall looks worse. The verifier's job, catching insufficient evidence, shows up in refusal behaviour, not recall at all.

So the question was settled by **progressive-removal ablation** (`python -m eval.ablation`): same questions, same retriever, same judge, varying only how much pipeline runs.

| Config | Calls | Cost/query | Latency | Evid. recall | Grounded | Correct | Behaviour |
|---|---|---|---|---|---|---|---|
| **A** retrieve→synthesize | 1.0 | **$0.0159** | **9.9s** | **1.00** | 0.98 | **1.00** | 1.00 |
| B + planner | 2.0 | $0.0207 | 13.9s | 0.88 | 0.99 | 0.97 | 1.00 |
| C + verifier/retry | 3.4 | $0.0490 | 15.6s | 0.88 | 0.98 | 0.98 | 1.00 |

The decision rule was **committed to before the run** (it is written into the script's docstring): *A ties or beats C → delete the stages; C wins on some questions → build a router; C wins everywhere → keep the pipeline.* A tied or won everywhere, so the stages are off.

**What the per-question deltas show.** C loses 0.50 evidence recall on Q1 and Q9, both deterministic, both caused by the planner rewriting the question into sub-queries that retrieve worse than the original. Q9 also loses 0.20 correctness, with a mechanistic explanation rather than a bare score gap: the planner dropped a document, so the answer had less to work with. Everywhere else, differences sit inside the judge's measured noise floor (±0.10 on borderline answers, see `eval/judge_variance.py`) and are reported as ties, not wins.

**The verifier was duplicating the synthesizer.** Behaviour-match stays 1.00 in config A: Q2 and Q3 are still correctly refused *without* a verifier, because the synthesizer prompt already carries a refusal rule. The verifier spent 44.8% of the pipeline's cost re-deciding something the next stage decides anyway.

**Where the money went.** Cost share and latency share are not the same stage, which is why both are tracked:

| Stage | % of LLM time | % of cost |
|---|---|---|
| synthesizer | 67.7% | 51.5% |
| planner | 20.0% | 3.7% |
| verifier | 12.3% | **44.8%** |

The verifier is cheap in *time* (one-word output) and expensive in *money* (it re-reads the entire evidence block as input). A latency-only view would have ranked it last and kept it.

**The system spent most on questions it could not answer.** Q2 and Q3, the two correct refusals, cost $0.146 and $0.122 against a $0.026 median, because the retry loop fires precisely when evidence looks insufficient. **52% of the benchmark's total cost went to 2 of 10 questions, both unanswerable.** Retrying cannot help when the corpus lacks the answer.

**Limits of this conclusion, stated plainly.** It holds for *this* corpus (733 mostly single-page articles where one retrieval already achieves recall@5 = 1.00 on these 10 questions, 0.94 across all 100) and *this* 10-question set, which contains no genuinely multi-hop questions. On a corpus with weak retrieval, the planner's ability to re-query is exactly the mechanism that would pay for itself, the literature's case for agentic RAG is that it repairs weak retrieval, and there is nothing here to repair. That is why the stages are disabled by configuration (`USE_PLANNER`, `USE_VERIFIER`) rather than deleted.

### Tool calling: the one stage that did earn its cost

The ablation above switched the planner and verifier off. It also predicted the
condition under which re-querying *would* pay: a corpus where retrieval is weak
enough to have something to repair. At n=100 that condition partly holds. Six
questions retrieve none of their expected documents, so `USE_TOOL_LOOP`
(`src/tools.py`) exposes retrieval to the model as a `search_docs` tool and lets
it run its own searches, reformulating when the passages come back unhelpful.

**The prediction was wrong, and the way it was wrong is the interesting part.**
Before writing the feature, the 6 failing questions were replayed through the
retriever with four hand-written reformulations each. Two recovered. That was
taken as a ceiling, on the reasoning that a model without hindsight would do
worse, and the feature was nearly abandoned on that basis.

| | Recovered | Cost/query |
|---|---|---|
| Hand-written reformulations (predicted ceiling) | 2 / 6 | n/a |
| **Model choosing its own searches** | **4 / 6** | **$0.0278** |
| Baseline, single retrieval | 0 / 6 | $0.0157 |

The model beat the hand-written ceiling because it reformulates *after* seeing
the results. The planner failed at n=10 for the mirror-image reason: it rewrote
the question up front, blind, and retrieved worse than the original. Same
mechanism, opposite outcome, and the difference is entirely whether the
reformulation is informed by feedback.

On a 20-question sample drawn from the questions that already retrieve correctly,
the tool loop lost **none** (20/20 retained), so the recovery is not being paid
for with regressions elsewhere. The model used 1 to 3 searches and never hit the
4-call cap.

**Why it is still off by default.** That measurement is retrieval-only: it asks
whether the expected document reached the model, not whether the answer got
better. Groundedness and correctness need the full judged run at n=100, which has
not been paid for yet. Flipping a default on unjudged evidence is the exact move
the rest of this section argues against, so the flag ships off and the evidence
for turning it on is written down rather than acted on. Zero regressions in 20
questions is also consistent with a real regression rate as high as ~14%, which
is another reason the wider run matters.

```bash
USE_TOOL_LOOP=true streamlit run app/main.py   # model runs its own searches
```

### A correction worth reading: how I got this wrong

An earlier version of this section claimed `recall@5 = 0.81` and "worst rank 8", concluding that `final_k=5` was truncating good results. **That was wrong, and the cause was my own diagnostic.** It measured retrieval with `top_k=50`, a candidate pool the agent never uses, on the assumption that a wider pool could only reveal more.

**RRF is not monotonic in pool size.** Its score is `Σ 1/(k + rank_i)`, so with `k=60`:

```
doc in BOTH lists at ranks 17 and 9  ->  1/77 + 1/69 = 0.0275
doc in ONE list at rank 2            ->  1/62         = 0.0161
```

A document that is mediocre in both retrievers **outranks one that is excellent in a single retriever.** At `top_k=10` the lists are short and cross-list overlap is rare, so a strong semantic-only match survives. At `top_k=50` many mutually-mediocre documents become visible and displace it. Widening the pool measurably *lost* the answer document for Q6 and Q9.

Two consequences:

1. **The standard "retrieve top-50, rerank to 5" recipe would degrade this system** as currently built. Widening the pool is only safe *together with* a reranker that repairs the ordering. The two are a pair, not independent upgrades. That is now the argument for the reranker, and it is a measured one.
2. **A diagnostic that does not mirror production is worse than no diagnostic.** It produced a confident, wrong conclusion about where the bottleneck was, and sent me at the wrong fix. The diagnostic now pins to `RETRIEVAL_TOP_K` and refuses to report depths beyond `RETRIEVAL_FINAL_K`.

Retrieval is **bit-identical across repeated runs** (verified across fresh clients with the BM25 index re-unpickled), so all of the above carries zero run-to-run noise and all end-to-end variance comes from the LLM planner.

A second signal points the same way: on **Q6**, the retriever alone scores recall@5 = 0.00 while end-to-end evidence recall is 1.00: the planner's query decomposition surfaced a document the raw query missed. That is the clearest case in the set of the agentic layer earning its cost.

**Honest caveats:**
- **The ablation is n=10.** Its CIs are wide by construction; one question moves any mean by ~10 points. The headline metrics are n=100, but the ablation that switched the planner and verifier off was not re-run at that size, so the pipeline-composition conclusion rests on the weaker sample. Re-running it at n=100 is the top priority.
- **Single judge, model-graded.** No human-labelled agreement (Cohen's κ) has been measured yet, so the judge itself is unvalidated.
- **Citation precision measures the weak thing** (see table above) and 1.00 should be read accordingly.
- **Latency ~19s** reflects 4-5 sequential LLM calls; fine for assisted lookup, too slow for live phone support.
- **Means hide the worst case.** The generated report lists worst-case rows for exactly this reason. Here, correctness bottoms out at 0.80 on Q9.

### Enhanced retrieval A/B: withdrawn pending re-measurement

An earlier A/B of stronger embeddings (`BAAI/bge-small-en-v1.5`) plus the cross-encoder reranker reported a recall gain of 0.78 → 0.89. **That comparison is withdrawn.** It was produced under the old harness: self-judged, with the mislabeled ground truth described below, and using the metric definition since renamed. It also rested on a single-question difference at n=10, well inside the noise floor. The eval set is now large enough (n=100) to resolve a difference of that size, so re-running it is no longer blocked, only unstarted.

The opt-in path still exists (it pulls in torch):

```bash
pip install -r requirements-enhanced.txt
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 RERANKER_ENABLED=true python -m src.retriever      # rebuild index
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 RERANKER_ENABLED=true python -m src.eval_harness
```

Contextual Retrieval and semantic chunking were deliberately **not** implemented: both target long multi-page documents and would cost real ingest-time API calls for little gain on a single-page corpus.

### On Q2 / Q3: a correction

Questions **2 (RAM for document-level processing)** and **3 (DB2 log disk space)** return refusals, and both refusals are **correct**. But an earlier version of this README justified that conclusion with a claim that was false, and the correction is more instructive than the original claim:

> ~~"The harness resolved it: for Q2 the relevant requirements docs *were* retrieved (recall@k=1.0, ~38 chunks)"~~

The harness recorded Q2 recall as **0.00**, not 1.00. The linked `metrics.json` contradicted the README. Worse, the ground truth listed `aiw0appservrq.pdf` as Q2's expected source; that file is a **conceptual "Application server" overview, not a hardware-requirements document**, so it could never have contained the answer. A mislabeled expected source was making a correct refusal look like a retrieval miss.

**What actually settles it** is a full-corpus scan, not the harness, and by construction the harness *cannot* settle it, since it only ever sees the top-k it retrieved. `python -m eval.verify_unanswerable` reads all 733 PDFs and reports the evidence:

- **Q2:** the only `GB` figure in the entire corpus is a log-file size limit in `pdfw_c_userprefs.pdf`. No RAM specification exists anywhere. Refusal correct.
- **Q3:** six documents mention DB2 (shutdown commands, Rocky Linux support, version notes, PostgreSQL coexistence). None states a log disk-space figure. Refusal correct.

The script exits non-zero if any candidate quantity appears, so it can gate CI. Q2's `expected_sources` is now `[]`, the truthful encoding of "no document can answer this", which correctly makes recall unmeasurable rather than zero.

**Why this is in the README rather than quietly fixed:** the failure mode here, a confident "verified" claim that the cited artifact did not support, is the single most common way eval numbers become untrustworthy, and it happened in a project whose stated selling point was honest measurement. The mislabeled ground-truth entry is the more interesting half: it penalised the system for a label error, and it was only visible by reading the source documents.

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
- Table-heavy content may have reduced retrieval accuracy due to PDF text extraction limitations.
- LLM API latency (~10-15s) may be too slow for live phone support - could be improved with smaller/local models.

---

## 9. Tech stack

| Category | Technology |
|---|---|
| **Language** | Python 3.11+ (developed on 3.13) |
| **PDF Parsing** | PyMuPDF 1.25.3 |
| **Vector Database** | ChromaDB 0.6.3 (local, all-MiniLM-L6-v2) |
| **Keyword Search** | rank_bm25 0.2.2 |
| **Agentic Framework** | LangGraph 0.2.74 |
| **LLM** | Claude Sonnet via langchain-anthropic 0.3.12 |
| **UI** | Streamlit 1.42.0 |
| **Configuration** | python-dotenv 1.0.1 |

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
Results and methodology are in [§7 Evaluation & Metrics](#7-evaluation-and-metrics).

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

## 12. Optional: cross-encoder reranker

A query-aware cross-encoder reranker (off by default to keep the base torch-free) can be enabled for higher precision@k:

```bash
pip install -r requirements-reranker.txt
RERANKER_ENABLED=true streamlit run app/main.py
```

It fuses a larger candidate pool, re-scores with `cross-encoder/ms-marco-MiniLM-L-6-v2`, and trims to the top-k. Lazy-loaded and cached, so the default lightweight path is untouched.

## 13. Deployment

Beyond the local Ngrok demo, the project ships a reproducible container path. See **[DEPLOYMENT.md](DEPLOYMENT.md)** for Docker, Render (one-click `render.yaml`), and Streamlit Cloud instructions.

```bash
docker build -t citera .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... -v "$PWD/data:/app/data" citera
```

## 14. Project status: demo vs production

Being straight about what this is:

**Built and working:** hybrid retrieval + RRF, agentic verify-retry loop, grounded/cited generation with refusal, multi-lingual answers, Glass Box UI, a quality eval harness, a unit-test suite + CI, and a containerised deploy path.

**Deliberately out of scope (next steps for true production):** auth in front of the app, secrets management, LLM-call caching + rate-limit/retry handling, request tracing (LangSmith), multi-turn conversation memory, and a CI-built index artifact instead of ingest-on-boot. These are tracked in [DEPLOYMENT.md](DEPLOYMENT.md).

## 15. Repository structure

```
Ricoh/
├── app/
│   └── main.py                  # Streamlit Glass Box dashboard
├── data/
│   └── *.pdf                    # Ricoh RPD docs, 733 PDFs (gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py                # Centralised configuration
│   ├── ingest.py                # PDF parsing + chunking pipeline
│   ├── retriever.py             # Hybrid retrieval (ChromaDB + BM25 + RRF + optional reranker)
│   ├── llm_factory.py           # LLM provider abstraction
│   ├── agent.py                 # LangGraph agentic state machine
│   ├── evaluate.py              # Latency/citation smoke test
│   └── eval_harness.py          # Quality eval harness (evidence recall, retriever recall@N, groundedness)
├── eval/
│   ├── ground_truth.json        # Curated expected answers/sources
│   ├── metrics.json             # Harness output, default config (generated)
│   ├── metrics_enhanced.json    # Harness output, bge + reranker A/B (generated)
│   ├── eval_report.md           # Harness output (generated)
│   └── verify_unanswerable.py   # Audits the "refuse" labels against the full corpus
├── tests/                       # pytest suite (offline, LLM mocked)
│   ├── test_ingest.py
│   ├── test_retriever.py
│   ├── test_agent.py
│   └── test_eval_metrics.py
├── .github/workflows/ci.yml     # GitHub Actions CI
├── notebooks/                   # Exploration notebooks
├── chroma_db/                   # Persisted ChromaDB + BM25 index (gitignored)
├── Dockerfile                   # Container build
├── .dockerignore
├── render.yaml                  # Render.com one-click deploy blueprint
├── DEPLOYMENT.md                # Docker / Render / Streamlit Cloud guide
├── .env.example                 # Copy to .env and fill in
├── .gitignore
├── requirements.txt             # Runtime deps
├── requirements-dev.txt         # + pytest (CI)
├── requirements-reranker.txt    # Optional cross-encoder extra
├── requirements-enhanced.txt    # Optional stronger embeddings + reranker
├── LICENSE                      # MIT
├── evaluation_results.csv       # Smoke-test output
├── evaluation_report.md         # Smoke-test output
└── README.md                    # This file
```

---

## 16. Roadmap

Ordered by what most improves the system, not by what is easiest to demo.

| Priority | Work | Why |
|---|---|---|
| **1** | Re-run the A/B/C ablation at n=100 | ~~Expand the eval set to 100+ questions with a held-out slice~~ **done**: the benchmark is now 100 questions with a 70/30 split. The ablation itself is still n=10, so the conclusion that switched two stages off is the weakest-powered claim left in the project. |
| **2** | Judge calibration: hand-label ~50, report Cohen's κ | The judge's noise floor is measured (0.00-0.10) but its *agreement with humans* is not. Chance-corrected, not raw. |
| **3** | Widen candidate pool (top-10 → top-50) + rerank to 5 | Worst observed rank is 8, so the ranking headroom is provable, not hoped-for. |
| ~~4~~ | ~~Adaptive routing~~ | **Done differently.** The ablation showed no question benefited from the agentic path, so routing was unnecessary. The stages were switched off outright. A router would have added a classifier to choose between a better option and a worse one. |
| **5** | Strip print-to-PDF boilerplate at ingest | 75% of chunks carry an identical header/breadcrumb (~4-6% of words). Low-risk cleanup. |
| **6** | Claim→span attribution instead of filename matching | Current citation precision only catches fabricated filenames. |
| **7** | Tracing, per-request cost/latency budgets, index built in CI | Production surface. |

**Deliberately deferred:** multi-lingual answering is currently a liability rather than a feature. The refusal marker is English-only, so a translated-only refusal would be scored as an answer. The synthesizer now pins the English canonical sentence to keep the eval sound, but full language support needs a language-aware detector before it is worth advertising.
