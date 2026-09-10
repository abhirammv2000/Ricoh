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

It started as an agentic pipeline (plan, retrieve, verify, retry) and ended up as a single retrieval call, because that is what the measurements supported. An ablation re-run at n=100 walked part of that back: the verifier still earns nothing, but the planner does help retrieval on the dev split, just not on the held-out one, so it stays behind a flag rather than deleted (see [§7](#7-evaluation-and-metrics)).

**Measured** with `claude-opus-5` judging `claude-sonnet-4-6` over the full corpus, methodology and caveats in [§7](#7-evaluation-and-metrics):

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

The verifier (C) adds no evidence recall over B and is slightly worse on the judged metrics, so it stays off. The planner (B) takes evidence recall from 0.94 to 1.00 on dev, but on the held-out 30 questions A and B score an identical 0.93, so the benefit does not replicate. Config A is the default: the planner's edge is real on one split, gone on the other, and not worth 1.7x the cost per query on every question. It stays behind `USE_PLANNER` for corpora where retrieval is weaker. Full per-split numbers and the earlier n=10 run in [§7](#7-evaluation-and-metrics).

Later work held up under harder tests: a judged ablation on a hand-built **multi-hop** set (the case the 100-question set was missing) still puts the planner's gain inside the judge noise floor; the **holdout** ablation, once judged, confirmed the dev-only pattern; a **cross-provider bakeoff** found `gemini-3.6-flash` holds answer quality at 1/50th the cost per query while `gpt-4o-mini` drops correctness 0.21; and **multi-turn** follow-up handling (history-aware query rewriting) answers follow-ups about as well as cold questions. All in [§7](#7-evaluation-and-metrics).

Brackets are 95% percentile-bootstrap confidence intervals.

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

Anthropic *prompt* caching is a different lever and was assessed as not
applicable here: every static prompt prefix in this system (the synthesizer
instructions, the judge rubric, the tool-loop system prompt) is a few hundred
tokens, well under the 1024-token minimum a cache breakpoint needs, and
everything above that minimum, the retrieved evidence, is different on every
request. Manufacturing a cacheable prefix would mean prepending ~800 tokens of
few-shot examples to the synthesizer, which is a prompt change that would
invalidate every headline number in section 7 and require re-running the
benchmark, for a saving of a few hundred cached input tokens per call. Not
worth it. The pricing table still models `cache_read` and `cache_write` so the
door is open if a future prompt grows a large reusable prefix for another
reason.

The dashboard also drills into one past request: pick a trace and see its
per-stage cost and, for the retrieval span, exactly which chunks fed the answer
at what rank and RRF score. That view was CLI-only (`src/trace_view.py`); it and
the dashboard now share `perf.format_trace` so they cannot disagree.

**Honest gap:** production traffic is not sampled back into the eval set.

---

## 1. Problem statement

Build a technical-support assistant that answers complex, multi-part questions about RICOH ProcessDirector using **only** the supplied documentation, and that refuses rather than guesses when the documentation does not contain the answer.

Field technicians and support engineers lose significant time searching documentation for specific procedures, error-code resolutions, and configuration steps. The system needs to ingest that documentation, understand natural-language questions, retrieve the relevant passages, and generate accurate, cited answers with zero hallucination.

The thing that actually shapes the design is the corpus: 733 individual help-topic articles, most of them a single page, not a handful of long manuals (median chunk: 307 words; 1,314 chunks total). So the core retrieval task is **selecting the right document out of 733**, not locating a passage within a long document. Almost every design decision below follows from that, see [§5](#5-data-handling-and-preprocessing).

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
3. **Runs** a LangGraph state machine whose composition is set by measurement: the Planner and Verifier stages exist but are **off by default**, because an ablation showed the verifier earns nothing and the planner's help does not replicate across splits ([§7](#7-evaluation-and-metrics)).
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
- **Agentic loop, built, measured, and switched off by default.** The rationale was that single-pass retrieval misses evidence on multi-part questions and a plan-then-verify loop catches the gaps. At n=100 the verifier still earns nothing, and the planner helps retrieval on the dev split but not the held-out one, so the loop is not worth 1.7x the cost per query here. It stays behind config flags for a corpus where retrieval is weaker ([§7](#7-evaluation-and-metrics)).

---

## 5. Data handling and preprocessing

- Dataset: official Ricoh ProcessDirector (RPD) documentation, 733 PDFs (~223 MB), stored in `data/` (gitignored due to size). Honest note: these are **individual help-topic articles**, most of which are a *single page* each, rather than a handful of 100+ page manuals. This is why nearly every citation reads "Page 1", and it means the real retrieval challenge here is **picking the right document out of 733**, not pinpointing a page within a long manual. The page-level citation machinery still works (and would matter for true multi-page manuals), but we call out the corpus shape rather than overstate it.
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
- **Other providers:** `src/llm_factory.py` wires OpenAI and Gemini (the latter via its OpenAI-compatible endpoint) behind the same interface; `LLM_PROVIDER=google pip install -r requirements-providers.txt` and a `GEMINI_API_KEY` is all it takes to run the agent on Gemini. `eval/provider_bakeoff.py` compares them against a shared retrieval and one fixed judge. Judged result ([§7](#7-evaluation-and-metrics)): `gemini-3.6-flash` holds answer quality at 1/50th the cost per query, `gpt-4o-mini` drops correctness 0.21. So the model matters, not just the price. The judge is pinned to Anthropic regardless of the agent provider, which actually makes the agent/judge pairing *more* independent when the agent is not Claude.

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

**2. Quality harness** (`src/eval_harness.py` -> `eval/metrics.json`, `eval/eval_report.md`), **the one to look at.**

| Metric | LLM-judged? | What it actually measures, and what it does *not* |
|---|---|---|
| Evidence recall | No | Did the expected document reach the synthesizer across all passes and retries? Deliberately *not* called recall@k: the accumulated evidence ranges from 5 to 40+ chunks, so calling it "@k" (k=5) would flatter the retriever. |
| Retriever recall@1/5/20 | No | The retriever in isolation, raw question, one pass, no planner. Comparing against evidence recall separates a *retrieval* failure from a *planning* failure. |
| Citation precision | No | Are cited documents present in the evidence? Catches fabricated filenames only, it does *not* verify that the cited document supports the claim. Real attribution correctness needs claim->span checking (not yet built). |
| Groundedness | Yes | Is every claim supported by the retrieved evidence? |
| Answer correctness | Yes | Does the answer convey the expected key facts, or refuse correctly? |
| Behaviour match | No | Did it answer when it should and refuse only when it should? |

```bash
python -m src.eval_harness            # full quality run (uses the LLM judge)
python -m src.eval_harness --no-judge # objective metrics only, no API calls
python -m eval.verify_unanswerable    # audit the "refuse" labels against the corpus
python -m eval.judge_variance         # measure the judge's own noise floor
```

### Why the judge is a different model

The judge runs on `claude-opus-5` while the agent runs on `claude-sonnet-4-6` (`JUDGE_MODEL` in `src/config.py`). This is not incidental: an earlier version of this harness used the **same model as both agent and judge**, which is the weakest possible configuration: a model grading its own output exhibits self-preference bias, and published work on LLM judges finds raw agreement badly overstates chance-corrected agreement.

Switching to an independent, stronger judge moved groundedness 0.977 -> 0.98 and correctness 0.97 -> 0.98.

**That difference means nothing, and it is worth being precise about why.** `python -m eval.judge_variance` scores an identical answer against identical evidence five times and reports the spread:

| Case | Groundedness across 5 identical runs | Spread |
|---|---|---|
| Q8 (unambiguous) | 1.00, 1.00, 1.00, 1.00, 1.00 | 0.00 |
| Q9 (borderline) | 0.60, 0.55, 0.55, 0.60, 0.55 | 0.05 |
| Q7 (borderline) | 0.65, 0.70, 0.70, 0.60, 0.70 | 0.10 |

The judge is perfectly stable on clear-cut answers and jitters by up to 0.10 on borderline ones, and borderline answers are precisely the ones that move an aggregate. A 0.003 shift in the mean is **more than an order of magnitude below the instrument's own noise floor**, before even accounting for agent nondeterminism producing different answers between runs.

So the honest conclusion is: **this eval cannot detect whether self-preference was inflating the scores.** The judge was changed because grading your own output is methodologically indefensible regardless of what the number does, not because the change was shown to matter. An earlier draft of this README claimed the scores "barely moved, so self-preference was not materially inflating them". That was a confounded comparison (different ground truth *and* different agent outputs between the two runs) asserting a conclusion the data cannot support.

**The general rule this establishes:** no claim about a metric change on this benchmark is credible until the change exceeds the judge's measured noise floor. It is why the eval set was grown to 100 and why the ablation was re-run at that size.

### An independent check: RAGAS faithfulness

The groundedness number still rests on one hand-rolled judge. `eval/ragas_eval.py` adds a third angle (after judge-noise and cross-judge): [RAGAS](https://docs.ragas.io), a widely used RAG-eval library, computing **faithfulness**, its analog of groundedness, with its own prompts and its own decomposition of the answer into atomic claims. RAGAS pins langchain/langgraph to 1.x and cannot share this project's env, so it runs in a throwaway virtualenv against a JSONL exported by `eval/ragas_export.py`; nothing in the RAGAS step imports `src`.

On 50 questions sampled from the n=100 run (`claude-sonnet-4-6` for RAGAS, against the harness's `claude-opus-5`):

| | mean |
|---|---|
| RAGAS faithfulness | 0.937 |
| Our groundedness judge | 0.971 |

RAGAS runs about 0.03 lower. The seven disagreements are all answers RAGAS scored 0.67 to 0.93 while the judge scored 0.75 to 1.00, RAGAS penalising a partially-supported claim harder because it decomposes the answer and scores claims one at a time. **The chance-corrected agreement is uninformative** (kappa -0.04): both raters score nearly everything acceptable, so there is no room for kappa to move, the same base-rate trap `eval/label_for_kappa.py` documents. So this neither validates the judge nor contradicts it. It shows the scores are not wildly off and the judge is not scoring in a way a second framework finds bizarre; only human labels close the real gap. Full output in [eval/ragas_results.md](eval/ragas_results.md).

### Measured results

Full corpus (733 PDFs -> 1,314 chunks), agent `claude-sonnet-4-6`, judge `claude-opus-5`, **100 questions** with a 70/30 dev/holdout split. Generated by `src/eval_harness.py` -> [eval/eval_report_n100.md](eval/eval_report_n100.md) / [eval/metrics_n100.json](eval/metrics_n100.json). Brackets are 95% percentile-bootstrap CIs:

| Metric | Mean | 95% CI | n |
|---|---|---|---|
| Behaviour-match rate | 0.98 | n/a | 100 |
| Groundedness | 0.96 | [0.95, 0.98] | 100 |
| Answer correctness | 0.97 | [0.94, 0.99] | 100 |
| Citation precision | 1.00 | [1.00, 1.00] | 100 |
| Evidence recall | 0.94 | [0.89, 0.98] | 100 |
| Mean latency | 10.1s | max 24.5s | 100 |

The earlier 10-question run reported 1.00 behaviour match, 0.98 groundedness and 0.98 correctness at 18.8s. Those numbers are **superseded, not deleted**: they are what a small sample looks like when it flatters you. Growing the set moved groundedness down 0.02, cut the intervals roughly in half, and surfaced the behaviour-match failure that 10 questions could not see. The old run is preserved in [eval/eval_report.md](eval/eval_report.md) / [eval/metrics.json](eval/metrics.json).

**The retriever in isolation.** This is the most actionable result in the project:

| Depth | Recall |
|---|---|
| recall@1 | 0.78 |
| recall@3 | 0.89 |
| recall@5 | 0.94 |

At **production settings** (`top_k=10, final_k=5`), retrieval on the raw question reaches 0.94 recall@5 across all 100 questions, so the retriever is not the main bottleneck. The pipeline wrapped around it was:

| Path (n=10 ablation) | Recall | LLM calls |
|---|---|---|
| Raw question, single retrieval | 1.00 (8/8) | 0 |
| Full agentic pipeline | 0.88 | ~4 |

At n=10 the planner degraded Q1 and Q9 by rewriting the question into sub-queries that retrieved worse than the original, and improved none. That pointed toward deleting the agentic layer. The n=100 re-run below shows it was a two-question artifact: at n=100 the planner does help retrieval on the dev split. Read the next section, not this line, for the current picture.

### The ablation (n=10)

This is the run that first set the default. It was later re-run at n=100, which
revised the planner half of the conclusion; that follow-up is the next section.

Retrieval recall is not answer quality, so "the planner hurts retrieval" was not sufficient grounds to remove it. The agent gathers *more* evidence on some questions (Q4: 17 chunks, Q7: 8), and extra context could plausibly raise groundedness even while document recall looks worse. The verifier's job, catching insufficient evidence, shows up in refusal behaviour, not recall at all.

So the question was settled by **progressive-removal ablation** (`python -m eval.ablation`): same questions, same retriever, same judge, varying only how much pipeline runs.

| Config | Calls | Cost/query | Latency | Evid. recall | Grounded | Correct | Behaviour |
|---|---|---|---|---|---|---|---|
| A retrieve->synthesize | 1.0 | $0.0159 | 9.9s | 1.00 | 0.98 | 1.00 | 1.00 |
| B + planner | 2.0 | $0.0207 | 13.9s | 0.88 | 0.99 | 0.97 | 1.00 |
| C + verifier/retry | 3.4 | $0.0490 | 15.6s | 0.88 | 0.98 | 0.98 | 1.00 |

The decision rule was **committed to before the run** (it is written into the script's docstring): *A ties or beats C -> delete the stages; C wins on some questions -> build a router; C wins everywhere -> keep the pipeline.* A tied or won everywhere, so the stages are off.

**What the per-question deltas show.** C loses 0.50 evidence recall on Q1 and Q9, both deterministic, both caused by the planner rewriting the question into sub-queries that retrieve worse than the original. Q9 also loses 0.20 correctness, with a mechanistic explanation rather than a bare score gap: the planner dropped a document, so the answer had less to work with. Everywhere else, differences sit inside the judge's measured noise floor (±0.10 on borderline answers, see `eval/judge_variance.py`) and are reported as ties, not wins.

**The verifier was duplicating the synthesizer.** Behaviour-match stays 1.00 in config A: Q2 and Q3 are still correctly refused *without* a verifier, because the synthesizer prompt already carries a refusal rule. The verifier spent 44.8% of the pipeline's cost re-deciding something the next stage decides anyway.

**Where the money went.** Cost share and latency share are not the same stage, which is why both are tracked:

| Stage | % of LLM time | % of cost |
|---|---|---|
| synthesizer | 67.7% | 51.5% |
| planner | 20.0% | 3.7% |
| verifier | 12.3% | 44.8% |

The verifier is cheap in *time* (one-word output) and expensive in *money* (it re-reads the entire evidence block as input). A latency-only view would have ranked it last and kept it.

**The system spent most on questions it could not answer.** Q2 and Q3, the two correct refusals, cost $0.146 and $0.122 against a $0.026 median, because the retry loop fires precisely when evidence looks insufficient. **52% of the benchmark's total cost went to 2 of 10 questions, both unanswerable.** Retrying cannot help when the corpus lacks the answer.

**Limits of this conclusion, stated plainly.** It holds for *this* corpus (733 mostly single-page articles where one retrieval already achieves recall@5 = 1.00 on these 10 questions, 0.94 across all 100) and *this* 10-question set, which contains no genuinely multi-hop questions. On a corpus with weak retrieval, the planner's ability to re-query is exactly the mechanism that would pay for itself, the literature's case for agentic RAG is that it repairs weak retrieval, and there is nothing here to repair. That is why the stages are disabled by configuration (`USE_PLANNER`, `USE_VERIFIER`) rather than deleted.

### The ablation, re-run at n=100

The n=10 run above concluded that the planner *hurt* retrieval (evidence recall 1.00 for config A vs 0.88 for B and C) and switched it off. That was the weakest-powered claim in the project, so `python -m eval.ablation --n100` re-ran it against the 100-question set: dev (70) with the judge, holdout (30) on the objective metrics only.

**Dev split (70 questions, judged):**

| Config | Calls/q | Cost/q | Latency | Evidence recall | Grounded | Correct | Behaviour |
|---|---|---|---|---|---|---|---|
| A retrieve -> synthesize | 1.0 | $0.0156 | 10.2s | 0.943 | 0.963 | 0.972 | 0.986 |
| B + planner | 2.0 | $0.0262 | 14.3s | **1.000** | 0.977 | 0.994 | 1.000 |
| C + verifier/retry | 3.0 | $0.0414 | 16.5s | 1.000 | 0.972 | 0.989 | 1.000 |

**Holdout split (30 questions, now judged):**

| Config | Evidence recall | Grounded | Correct | Behaviour |
|---|---|---|---|---|
| A | 0.933 | 0.963 | 0.968 | 1.000 |
| B | 0.933 | 0.972 | 0.982 | 1.000 |

Two things changed and one did not.

**The planner is not harmful.** The n=10 finding that it rewrote questions into worse sub-queries was two questions out of ten and did not generalise. At n=100 the planner *helps* on dev, taking evidence recall from 0.943 to 1.000, which is the objective no-judge metric, so it is four real questions where A's single retrieval missed the document and B's sub-queries found it.

**But the benefit does not replicate.** On the held-out 30, A and B score an identical 0.933 evidence recall with zero per-question differences. The judged run added later confirms it: groundedness moves 0.963 -> 0.972 and correctness 0.968 -> 0.982, both inside the judge's ~0.10 noise floor. A planner with a general benefit should show it on both splits. It shows it on one. That is the signature of a small effect that is inside sampling noise at n=30 to 70, not a reliable win.

**The verifier still earns nothing.** Config C matches B on evidence recall (both 1.000) and is slightly *lower* on grounded and correct. Same as at n=10.

So config A stays the default. The honest change is to the reasoning: the planner is roughly neutral here rather than harmful, and it stays behind `USE_PLANNER` for a corpus where retrieval is weak enough for its re-querying to matter. The correctness and behaviour gaps on dev (0.972 -> 0.994, 0.986 -> 1.000) are inside the judge's ±0.10 noise floor and are not load-bearing.

Raw per-config dumps are in `eval/ablation/generated_questions_dev/` (`--n100`) and `_holdout/` (`--split holdout --configs A B`).

### The ablation on a multi-hop set

Every question in the 100-set is effectively single-hop: `eval/verify_multihop.py` shows one retrieval already reaches every required document on 92 of them. The planner decomposes a question into sub-queries, so it can only help where one query is not enough. `eval/multihop_questions.json` is a hand-written 20-question set of two-document questions (8 of which a single retrieval demonstrably misses one required document). This is the slice where the planner should earn its cost.

Judged A/B/C on those 20 (`python -m eval.ablation --ground-truth eval/multihop_questions.json`):

| Config | Calls | Cost/q | Latency | Evidence recall | Grounded | Correct | Behaviour |
|---|---|---|---|---|---|---|---|
| A retrieve -> synthesize | 1.0 | $0.0205 | 16.0s | 0.775 | 0.973 | 0.909 | 1.000 |
| B + planner | 2.0 | $0.0317 | 25.4s | 0.825 | 0.980 | 0.919 | 1.000 |
| C + verifier/retry | 3.0 | $0.0485 | 26.3s | 0.825 | 0.979 | 0.911 | 1.000 |

**The planner has a real mechanism, and it fires rarely.** It recovered a missed document on exactly two questions (Q5 and Q14, both 0.50 -> 1.00 evidence recall) by splitting "what does feature X do *and* how do I configure it" into two searches. On the other seven questions where config A missed a document, decomposition did not help. So evidence recall goes 0.78 -> 0.82: four real questions' worth, all of it on the slice built to be hardest, at 1.5x the cost per query on every question.

**It does not reach the answer.** Groundedness and correctness move by less than the judge's 0.10 noise floor (0.909 -> 0.919 correct). The recovered evidence on Q5 and Q14 did not change what the judge saw.

**The verifier earns nothing here either.** Config C matches B on evidence recall and is inside noise on the judged metrics, at another 1.5x cost. Same result as n=10 and n=100.

**The real bottleneck on multi-hop questions is synthesis, not retrieval or planning.** Config A correctness drops to 0.909 (from ~0.97 on the single-hop set), and bottoms out at 0.55 on Q12 and 0.60 on Q20, both questions where retrieval found evidence for both halves but the answer got one half thin or wrong. Neither the planner nor the verifier addresses "combine two retrieved facts correctly", so neither closes that gap.

Config A stays the default. The planner stays behind `USE_PLANNER` with a sharper rationale than before: on this corpus its re-query mechanism only fires on about 10% of even the multi-hop questions, and when it does fire the extra evidence does not change the answer. Per-config reports in `eval/ablation/multihop_questions/`.

### Tool calling

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
| Model choosing its own searches | 4 / 6 | $0.0278 |
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

### Routing: escalate only what needs it

The tool loop helps a handful of questions and costs 1.8x on all of them, so
running it unconditionally is the wrong trade. `src/router.py` runs it
selectively.

The first design read a confidence signal off the one retrieval (top RRF score,
rank-1 to rank-2 margin, how many distinct documents are in the top 5) and
escalated when it looked weak. `python -m eval.calibrate_router` measured whether
any of those separate the questions that missed from the ones that hit. On the
dev split they do not: for `top_rrf` the hits span `0.0164` to `0.0328` and the
misses span `0.0164` to `0.0323`, so any cutoff that catches all 4 misses also
escalates 27 of the 66 questions that were already fine. That signal is a dead
end on this corpus and the calibration script records why.

What the router actually does is escalate after the fact: run the cheap path,
and if the synthesizer emits the refusal marker, hand the question to the tool
loop and take that answer. The trigger is the model's own "I cannot answer this
from the evidence", which is a call already paid for, so the confident majority
costs exactly one call.

Judged on the dev 70 (`USE_ROUTER=true python -m src.eval_harness --ground-truth
eval/generated_questions.json --split dev`), it escalated exactly one question.
Against plain config A it moved behaviour 0.986 -> 1.000, evidence recall 0.943
-> 0.957 and correctness 0.972 -> 0.980, for +0.04 calls and +$0.001 per query.
Real, but with one escalation most of that is run-to-run variance. The router is
a cheap safety net for the occasional wrong refusal, not a quality lever; the
planner (config B above) is the lever, and it still beats the router on evidence
recall (1.000 vs 0.957). Off by default (`USE_ROUTER`).

### Multi-turn follow-ups

Every question above is asked cold. Real support conversations are not: "what
inserter brands does RICOH ProcessDirector support?" is followed by "how do I
create a controller object if there isn't one for mine?", and the follow-up
retrieves nothing useful because it never names inserters.

`src/conversation.py` handles this the standard way. Before retrieval, a follow-up
is rewritten into a standalone question using the last few turns as context, so
"do those need a serial number?" becomes "does an Intelligent Mail barcode need
a serial number?" and then goes through the exact same pipeline as any other
question.

Two things keep this from undermining the rest of this section. The synthesizer
still answers only from retrieved evidence, so a bad rewrite degrades to a
retrieval miss and usually a refusal, not a hallucination. And the single-turn
path is byte-for-byte unchanged: with no history there is no rewrite call, so
the numbers above still describe what a one-shot question does.

**How well it works.** `eval/multiturn_questions.json` is 12 conversation chains
(36 turns); `eval/multiturn_eval.py` walks each chain carrying the real prior
answers as history and judges every turn against the hand-written standalone
intent.

*The rewrite is good.* Mean cosine to the hand-written standalone target is
**0.92** across the 24 follow-ups. "do those need a serial number?" became "does
an Intelligent Mail barcode need a serial number?", "what about the application
servers?" became "what operating system do the application servers run on?".

*The retrieval lift is the point.* Retriever recall on the raw follow-up is
**0.63**; after condensation it is **0.88**. Several follow-ups ("which step
template do I add for it?", "how would I track two deadlines?") retrieve nothing
on their own and land the right document once rewritten.

*Follow-ups are answered as well as cold questions.* Judged: groundedness 0.959
on follow-ups vs 0.983 on first turns, correctness 0.929 vs 0.944, behaviour
match 1.00 on both. Every gap is inside the judge's noise floor. Groundedness
holding is the load-bearing result: the system is not hallucinating on
follow-ups.

*Two honest wrinkles.* One chain (`custom-props`) still scores poorly, and its
first turn scores poorly too, so it is a retrieval/labeling weak spot unrelated
to multi-turn. And on two turns condensation *hurt* retrieval by adding a term
that pulled the query toward a broader overview document ("what data collectors
can I set up with it?" -> "...with the Reports feature?" retrieved the Reports
overview instead of the data-collector page). Over-specification is a real
failure mode of query rewriting, small here but worth naming. Per-turn table in
`eval/multiturn_report.md`.

### A diagnostic I got wrong

An earlier version of this section claimed `recall@5 = 0.81` and "worst rank 8", concluding that `final_k=5` was truncating good results. **That was wrong, and the cause was my own diagnostic.** It measured retrieval with `top_k=50`, a candidate pool the agent never uses, on the assumption that a wider pool could only reveal more.

**RRF is not monotonic in pool size.** Its score is `Σ 1/(k + rank_i)`, so with `k=60`:

```
doc in BOTH lists at ranks 17 and 9  ->  1/77 + 1/69 = 0.0275
doc in ONE list at rank 2            ->  1/62         = 0.0161
```

A document that is mediocre in both retrievers **outranks one that is excellent in a single retriever.** At `top_k=10` the lists are short and cross-list overlap is rare, so a strong semantic-only match survives. At `top_k=50` many mutually-mediocre documents become visible and displace it. Widening the pool measurably *lost* the answer document for Q6 and Q9.

Two consequences:

1. **The standard "retrieve top-50, rerank to 5" recipe would degrade this system** as currently built. Widening the pool is only safe *together with* a reranker that repairs the ordering. The two are a pair, not independent upgrades. The sweep confirms this: with the reranker on, `top_k=10` and `top_k=20` give identical recall, so the reranker does repair the wider pool ([Embedding model sweep](#embedding-model-sweep)).
2. **A diagnostic that does not mirror production is worse than no diagnostic.** It produced a confident, wrong conclusion about where the bottleneck was, and sent me at the wrong fix. The diagnostic now pins to `RETRIEVAL_TOP_K` and refuses to report depths beyond `RETRIEVAL_FINAL_K`.

Retrieval is **bit-identical across repeated runs** (verified across fresh clients with the BM25 index re-unpickled), so all of the above carries zero run-to-run noise and all end-to-end variance comes from the LLM planner.

A second signal points the same way: on **Q6**, the retriever alone scores recall@5 = 0.00 while end-to-end evidence recall is 1.00: the planner's query decomposition surfaced a document the raw query missed. That is the clearest case in the set of the agentic layer earning its cost.

**Honest caveats:**
- **The ablation split is dev-heavy.** It has been run at n=100 (see the section above), but the judged half is the 70-question dev split; holdout was measured on objective metrics only. The planner's dev-split edge on the judged metrics has not been confirmed with the judge on holdout.
- **Single judge, model-graded.** No human-labelled agreement (Cohen's κ) has been measured yet, so the judge itself is unvalidated. `eval/human_labels.json` is a prepared 30-item worksheet (passages included) waiting on the labelling pass. The RAGAS cross-check on 50 answers (above) is consistent with the judge but is still model-graded and cannot substitute for human labels.
- **Citation precision measures the weak thing** (see table above) and 1.00 should be read accordingly.
- **Latency ~10s** on the default single-call path; fine for assisted lookup, too slow for live phone support. The planner and verifier flags push it to 14 to 17s.
- **Means hide the worst case.** The generated report lists worst-case rows for exactly this reason. Here, correctness bottoms out at 0.80 on Q9.

### Embedding model sweep

The retriever runs on `all-MiniLM-L6-v2`, ChromaDB's built-in default and a 2021 model. An earlier A/B against `BAAI/bge-small-en-v1.5` plus the reranker reported a recall gain of 0.78 to 0.89. **That number is withdrawn:** it was self-judged, used the mislabeled ground truth described below, rested on a single question at n=10, and used a metric definition since renamed.

`eval/sweep_embeddings.py` replaces it with a retrieval-only sweep over the 100-question set: recall@1/3/5, MRR and nDCG@5 per (embedding model, candidate pool), on dev / holdout / all. No LLM, so it is exact and free. Each model gets its own index under `eval/indexes/`; the cross-encoder reranker is behind `--rerank` because its per-candidate transformer pass dominates the run.

At `top_k=10`, over all 100 questions:

| model | recall@1 | recall@3 | recall@5 | MRR | missed |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (current) | 0.78 | 0.89 | 0.94 | 0.85 | 6 |
| bge-small-en-v1.5 | 0.79 | 0.91 | 0.93 | 0.85 | 7 |

**The withdrawn gain does not reproduce at n=100.** bge-small is a wash: recall@3 is up 0.02, recall@5 is down 0.01, MRR is identical, and it misses one more question than MiniLM. It does not even fail on the same questions, it fixes one (Q83) and breaks two others (Q46, Q92). On the dev split bge-small's recall@1 looks better (0.80 vs 0.74); on holdout it looks worse (0.77 vs 0.87). A model whose ranking flips between splits is not a real improvement. The 2021 embedding model is not what is costing retrieval quality on this single-page corpus, so MiniLM stays.

The sweep also confirms the RRF non-monotonicity from the diagnostic below: widening the pool to `top_k=20` without a reranker moves MiniLM's recall@5 from 0.94 to 0.92 and its miss count from 6 to 8.

**The reranker (`--rerank`).** A `cross-encoder/ms-marco-MiniLM-L-6-v2` pass over the fused pool, all 100 questions:

| model | rerank | recall@1 | recall@3 | recall@5 | missed |
|---|---|---|---|---|---|
| MiniLM | no | 0.78 | 0.89 | 0.94 | 6 |
| MiniLM | yes | 0.77 | 0.95 | 0.97 | 3 |
| bge-small | yes | 0.75 | 0.95 | 0.97 | 3 |

The reranker recovers 3 of the 6 questions that retrieved nothing and takes recall@3 from 0.89 to 0.95. Three caveats keep it off by default:

1. **The gain is dev-only.** On dev, reranked recall@5 is 0.99; on the held-out 30 it is 0.93, exactly the un-reranked number, and reranked recall@1 on holdout *drops* from 0.87 to 0.67. Same split-dependent pattern as every other retrieval change in this section.
2. **It costs a transformer pass per query**, ~3 to 4s on CPU, which more than doubles the default path's latency.
3. **It makes the embedding choice moot.** MiniLM and bge-small converge to identical reranked numbers, because the cross-encoder is doing the ranking, not the embedder. `top_k=10` and `top_k=20` also converge once it is on, since it repairs the wider pool.

So the reranker is the clearest lever for the handful of hard questions if a latency budget allows, and it ships behind `RERANKER_ENABLED` with the evidence written down rather than turned on. bge-base was not built (a fresh torch pass over the corpus, slow without a GPU):

```bash
pip install -r requirements-enhanced.txt
python -m eval.sweep_embeddings --build bge-base
python -m eval.sweep_embeddings --measure --rerank
```

Contextual Retrieval and semantic chunking were deliberately **not** implemented: both target long multi-page documents and would cost real ingest-time API calls for little gain on a single-page corpus.

### Cross-provider bakeoff

The system runs on `claude-sonnet-4-6`. `src/llm_factory.py` also wires OpenAI and Gemini (the latter through its OpenAI-compatible endpoint), and `eval/provider_bakeoff.py` runs config A on each against a *shared, identical* retrieval and the same `claude-opus-5` judge, so any difference is the synthesizer model. On 20 questions from the dev split:

| model | cost/query | latency | out tokens | groundedness | correctness | behaviour |
|---|---|---|---|---|---|---|
| claude-sonnet-4-6 | $0.0151 | 9.9s | 476 | 0.980 | 0.993 | 1.00 |
| gemini-3.6-flash | $0.0003 | 5.2s | 240 | 0.998 | 0.953 | 1.00 |
| gpt-4o-mini | $0.0005 | 2.6s | 146 | 0.962 | 0.779 | 0.90 |

**Gemini Flash is a real cost lever; GPT-4o-mini is not.** Gemini holds groundedness (actually higher) and lands correctness 0.953 vs 0.993, a 0.04 gap inside the judge's ~0.10 noise floor, at **1/50th the cost per query** and roughly 2x faster. GPT-4o-mini drops correctness by 0.21, well outside noise, and refuses two questions it should have answered. So "swap to a cheap model" is not one decision, it depends which cheap model: on this task Gemini Flash would be a defensible production choice, `gpt-4o-mini` would be a downgrade. The Anthropic judge is held constant precisely so this comparison is not itself provider-biased.

### A correction on Q2 and Q3

Questions 2 (RAM for document-level processing) and 3 (DB2 log disk space) return refusals, and both refusals are correct. But an earlier version of this README justified that conclusion with a claim that was false, and the correction is more instructive than the original claim:

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
| Language | Python 3.11+ (developed on 3.13) |
| PDF Parsing | PyMuPDF 1.25.3 |
| Vector Database | ChromaDB 0.6.3 (local, all-MiniLM-L6-v2) |
| Keyword Search | rank_bm25 0.2.2 |
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

A second CI job is a **retrieval regression gate** (`python -m eval.ci_gate`): it runs real hybrid retrieval against the committed `demo_index/` on the seed questions and fails the build if recall@1/3/5 drops below `eval/ci_baseline.json`. The unit suite never touches a real index, so this is what would catch an RRF or fusion bug that still passes every mocked test. The same job also checks that `demo_index` has not drifted from the benchmark it serves: if `eval/ground_truth.json` gains a question whose expected document is not baked into the index, the build fails with a "rebuild demo_index" message. `demo_index` is a small curated slice, so the recall check is a smoke gate, not a quality measurement; the full-corpus numbers come from the paid harness in [§7](#7-evaluation-and-metrics).

## 12. Optional: cross-encoder reranker

A query-aware cross-encoder reranker (off by default to keep the base torch-free) can be enabled for higher precision@k:

```bash
pip install -r requirements-reranker.txt
RERANKER_ENABLED=true streamlit run app/main.py
```

It fuses a larger candidate pool, re-scores with `cross-encoder/ms-marco-MiniLM-L-6-v2`, and trims to the top-k. Lazy-loaded and cached, so the default lightweight path is untouched. Whether it earns its latency on this corpus is measured by `python -m eval.sweep_embeddings --measure --rerank` ([§7](#7-evaluation-and-metrics)).

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
│   ├── conversation.py          # History-aware follow-up rewriting for multi-turn
│   ├── router.py                # Cheap path, escalate to the tool loop on a refusal
│   ├── tools.py                 # Tool-calling retrieval loop (USE_TOOL_LOOP)
│   ├── guardrails.py            # Prompt-injection screen at the API edge
│   ├── instrumentation.py       # Per-stage cost / token / latency spans
│   ├── perf.py / trace_view.py  # Dashboard rollups and per-request drill-down
│   ├── semantic_cache.py        # Optional answer cache (off by default)
│   ├── evaluate.py              # Latency/citation smoke test
│   └── eval_harness.py          # Quality eval harness (evidence recall, retriever recall@N, groundedness)
├── eval/
│   ├── ground_truth.json        # Curated expected answers/sources
│   ├── generated_questions.json # The 100-question set, 70/30 dev/holdout
│   ├── metrics.json             # Harness output, default config (generated)
│   ├── eval_report_n100.md      # Harness output, 100 questions (generated)
│   ├── ablation.py              # Progressive-removal pipeline ablation
│   ├── ci_gate.py               # Retrieval regression gate (runs in CI vs demo_index)
│   ├── ragas_export.py          # Export a harness slice for the RAGAS cross-check
│   ├── ragas_eval.py            # RAGAS faithfulness vs our judge (separate venv)
│   ├── multihop_questions.json  # 20 hand-written two-document questions
│   ├── verify_multihop.py       # Confirms the multi-hop set stresses retrieval
│   ├── multiturn_questions.json # 12 conversation chains for the follow-up eval
│   ├── multiturn_eval.py        # Judged multi-turn evaluation
│   ├── provider_bakeoff.py      # Cross-provider synthesizer comparison
│   ├── run_paid_batch.sh        # The judged runs that need Anthropic credits
│   ├── sweep_embeddings.py      # Retrieval-only embedding-model comparison
│   ├── calibrate_router.py      # Whether a retrieval signal can drive the router
│   ├── label_for_kappa.py       # Judge-vs-human agreement worksheet + scoring
│   └── verify_unanswerable.py   # Audits the "refuse" labels against the full corpus
├── tests/                       # ~25 pytest modules, offline, LLM mocked
│   ├── test_ingest.py test_retriever.py test_agent.py test_router.py
│   ├── test_conversation.py test_tools.py test_guardrails.py
│   ├── test_eval_metrics.py test_ci_gate.py test_perf.py ...
│   └── (one per src/ and eval/ module worth guarding)
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

| Priority | Work | Status |
|---|---|---|
| 1 | Re-run the A/B/C ablation at n=100 | **Done, including the judge on holdout.** The n=10 "planner is harmful" finding did not hold: the planner helps evidence recall on dev, not on holdout; the judged holdout run confirms groundedness and correctness move inside the noise floor. Verifier earns nothing on any split. Config A stays default. [§7](#7-evaluation-and-metrics). |
| 2 | Judge calibration: hand-label 30, report Cohen's κ | Worksheet ready (`eval/human_labels.json`, passages included); a RAGAS faithfulness cross-check on 50 answers is consistent with the judge (0.937 vs 0.971, [§7](#7-evaluation-and-metrics)). The human labelling pass is the last open item on the eval side. |
| 3 | Better embedding model, wider pool, rerank | **Done.** bge-small does not beat MiniLM at n=100, so the withdrawn 0.78 -> 0.89 A/B does not reproduce and MiniLM stays. The reranker takes all-100 recall@5 from 0.94 to 0.97 and halves the miss count, but the gain is dev-only and it doubles latency, so it stays behind `RERANKER_ENABLED`. bge-base not built (no GPU). [§7](#7-evaluation-and-metrics). |
| 4 | Adaptive routing | **Done.** `src/router.py` escalates to the tool loop on a refusal (a pre-retrieval confidence signal was tried first and does not separate misses from hits). Judged on dev: escalates rarely, small gain over config A, within judge noise. Off by default (`USE_ROUTER`). |
| 5 | Strip print-to-PDF boilerplate at ingest | **Measured, not worth it.** Every page carries a PDF-export timestamp and an "N of M" line, but that is **1.5% of corpus words**, not the 4-6% first estimated, and both strings appear on 100% of pages so they carry zero BM25 IDF and shift every embedding identically: no measurable retrieval effect. Doing it would still force a budgeted re-eval to keep the headline numbers honest, for a sub-2% token saving. Left alone. |
| 6 | Claim->span attribution instead of filename matching | **Two free proxies tried, both reverted; RAGAS faithfulness is the working answer.** A MiniLM-cosine proxy and then a local NLI model (`nli-deberta-v3-base`) were each built and run over the 100 answers. Both produced numbers that contradict the judge's 0.96 groundedness (the NLI run flagged 8% of citations as "contradicted"). The cause is the synthesizer's answer style: it restructures sources into tables, worked examples and numbered steps, and answers non-English questions in the user's language, so sentence-level entailment against the raw chunk is not a fair test. Claim decomposition by an LLM handles that, which is what the RAGAS faithfulness cross-check does ([§7](#7-evaluation-and-metrics)). A deeper per-claim attribution metric still needs a dedicated judged pass. |
| 7 | Tracing, per-request cost/latency budgets, index built in CI | Tracing, per-request instrumentation and a per-request dashboard drill-down are done ([Observability](#observability)). A retrieval regression gate runs in CI against `demo_index`, and also fails if the index drifts from the benchmark it serves ([§11](#11-testing-and-ci)). A full-corpus index built in CI still needs the source PDFs it does not have; `demo_index` stays a committed artifact. |
| 8 | A genuinely multi-hop question set + ablation on it | **Done.** `eval/multihop_questions.json` is 20 hand-written two-document questions (8 of which one retrieval misses a required doc). Judged A/B/C: the planner takes evidence recall 0.78 -> 0.82, recovering a document on 2 of the 20, but grounded and correct stay inside the judge noise floor, and the verifier still earns nothing. The real bottleneck turned out to be synthesis (config A correctness 0.909, down from ~0.97 on single-hop), which no config addresses. Config A stays default. [§7](#7-evaluation-and-metrics). |
| 9 | Judged multi-turn conversation eval | **Done.** 12 chains, 36 turns, every turn judged. Condensation lifts follow-up retriever recall 0.63 -> 0.88 and the rewrites hit 0.92 cosine to the hand-written target. Follow-ups are answered as well as cold questions: groundedness 0.959 vs 0.983, correctness 0.929 vs 0.944, behaviour 1.00 on both, all inside noise. Also surfaced that condensation can over-specify and hurt retrieval on a couple of turns. [§7](#7-evaluation-and-metrics). |
| 10 | Cross-provider bakeoff | **Done.** `src/llm_factory.py` wires OpenAI and Gemini; `eval/provider_bakeoff.py` runs config A on each against a shared retrieval and one fixed opus judge. `gemini-3.6-flash` holds correctness (0.953 vs Sonnet 0.993, inside noise) at 1/50th the cost; `gpt-4o-mini` drops correctness 0.21 and refuses two questions it should answer. The cheap-model choice is model-specific, not just price. [§7](#7-evaluation-and-metrics). |

**Deliberately deferred:** multi-lingual answering is currently a liability rather than a feature. The refusal marker is English-only, so a translated-only refusal would be scored as an answer. The synthesizer now pins the English canonical sentence to keep the eval sound, but full language support needs a language-aware detector before it is worth advertising.
