# Citera Roadmap and Decision Log

**Goal:** a RAG system where every design choice is backed by a measurement.
Every design choice must be justifiable with evidence, and every number must be
reproducible from a command in this repo.

This document is the single source of truth for project state.

---

## Definition of done

The project is **finished** when all five of these are true. It is explicitly
*not* a goal to keep improving the system past this line.

- [x] **1. The evaluation is trustworthy.** Independent judge, honest metric
      names, confidence intervals, measured judge noise floor.
- [x] **2a. The evaluation has enough power.** 100 generated questions with a
      70/30 dev/holdout split, labels widened by adjudication, difficulty audited.
- [~] **2b. Judge validated.** Two model-based checks done (below); the human
      leg still needs ~15 labels: `python -m eval.label_for_kappa --sample 30`.
      Shippable without it, with the limitation stated.
- [x] **3. Cost and latency are measured per stage**, not estimated.
- [x] **4a. It is observable.** Request-level traces with per-stage spans and
      chunk attribution, persisted and queryable (`python -m src.trace_view`).
- [x] **4b. Deployable, verified locally.** Self-contained image builds, serves,
      and answers a real query; the final `git push` + Render click is manual.
- [ ] **5. Every constant and component is defensible.** No magic numbers, no
      claim the linked artifact does not support, no contradiction between the
      README and the code.

When these are checked, the project ships. Remaining ideas move to
"Explicitly out of scope" below rather than extending the timeline.

---

## Status

| Phase | Goal | Status |
|---|---|---|
| **0** | Make the evaluation trustworthy | Done |
| **1** | Eval set to ~100 questions + judge calibration | Set built; κ still needs ~15 human labels |
| **2** | Improve retrieval | Done, no work needed, see D-4 |
| **3** | Cut cost and latency | Done, see D-6 |
| **4** | Tracing + live deployment | Tracing done; image verified, deploy is manual |
| **5** | Final consistency pass, commit, ship | Last |

**Remaining: push and deploy to Render. Optionally ~15 human labels for κ.**

---

## Decision log

Each entry is a decision that changed the system, the evidence behind it, and
the command that reproduces that evidence.

### D-1 · The index could not build at all
`ChromaDB.get_or_create_collection(embedding_function=None)` **overrides** the
default rather than falling back to it, so every upsert raised and `chroma_db/`
stayed empty. Now the kwarg is omitted when unset.
-> `python -m src.retriever` · 1,314 chunks from 733 PDFs

### D-2 · The judge may not grade its own output
The harness used one model as both agent and judge: self-preference bias makes
the scores uninterpretable. Judge is now `claude-opus-5`, agent `claude-sonnet-4-6`.
The scores barely moved, but **that is not evidence the bias was absent**: the
change (0.003) is far below the judge's own noise floor.
-> `JUDGE_MODEL` in `src/config.py`

### D-3 · The judge's precision is measured, not assumed
Scoring an identical answer 5× gives **0.00 spread on clear-cut answers, up to
0.10 on borderline ones**. Any claimed improvement smaller than that is noise.
-> `python -m eval.judge_variance`

### D-4 · Retrieval needs no work: it is already perfect here
At production settings a single retrieval on the raw question finds **every
expected document (8/8, worst rank 5)**. The bottleneck was never retrieval.
-> `python -m eval.sweep_retrieval_depth`

### D-5 · Widening the candidate pool *hurts*: RRF is not monotonic
The standard "retrieve top-50, rerank to 5" recipe **degrades** this system
without a reranker. RRF scores `Σ 1/(k+rank)`, so a document mediocre in both
retrievers (`1/77 + 1/69 = 0.0275`) outranks one excellent in a single retriever
(`1/62 = 0.0161`). Widening surfaces mutually-mediocre documents that displace
strong semantic-only matches. Widening is only safe *paired with* a reranker.
-> `python -m eval.sweep_retrieval_depth`, `python -m eval.sweep_rrf_k`

### D-6 · The agentic pipeline was removed, and quality improved
Progressive-removal ablation, decision rule committed to before the run:

| Config | Calls | Cost/q | Latency | Evid. recall | Grounded | Correct |
|---|---|---|---|---|---|---|
| **A** retrieve->synthesize | **1.0** | **$0.0159** | **9.9s** | **1.00** | 0.98 | **1.00** |
| B + planner | 2.0 | $0.0207 | 13.9s | 0.88 | 0.99 | 0.97 |
| C + verifier/retry | 3.4 | $0.0490 | 15.6s | 0.88 | 0.98 | 0.98 |

A matched or beat C on every metric at 3.1× lower cost. The verifier was
duplicating the synthesizer's own refusal rule (behaviour-match stays 1.00
without it). **Scope:** this corpus, this question set. On a corpus with weak
retrieval the planner is the right mechanism, which is why the stages are
disabled by config (`USE_PLANNER`, `USE_VERIFIER`) rather than deleted.
-> `python -m eval.ablation`

### D-7 · `RRF_K = 60` kept, now by measurement not citation
Everything k≥5 achieves recall 1.00 (broad plateau); pure rank-driven fusion
(k=0,1) is measurably worse. Keeping the literature default is justified because
nothing in the data argues for moving it.
-> `python -m eval.sweep_rrf_k`

### D-8 · "Unanswerable" labels are verified against the whole corpus
The harness structurally cannot prove a refusal is correct: it only sees its own
top-k. A full-corpus scan can. Q2 and Q3 refusals confirmed correct; a *mislabeled
expected source* was found and fixed in the process.
-> `python -m eval.verify_unanswerable` (exits non-zero if a claim fails)

### D-9 · Cost and latency identify different bottlenecks
The verifier was **12.3% of latency but 44.8% of cost** (one-word output, re-reads
the full evidence block as input). A latency-only view would have kept it. Also:
**52% of benchmark cost went to the 2 unanswerable questions**, because the retry
loop fires exactly when retrying cannot help.
-> `src/instrumentation.py`, reported by `python -m src.eval_harness`

### D-10 · The benchmark now has enough power to be worth trusting
100 questions generated from sampled chunks, so `expected_sources` is known **by
construction**. Two guards against the usual generated-benchmark failures:

* **Too easy**: measured lexical overlap with the source chunk (68%) and recall
  headroom. recall@5 = 0.94, *below* ceiling, so the set can distinguish systems.
  The original 10-question set sat at 1.00 and could not.
* **Labels too narrow**: 47 questions initially had a different top hit. A strict
  adjudicator confirmed **25 of them were label errors**, not retrieval errors
  (many help topics answer the same question). Labels widened, each with a
  recorded justification. recall@1 rose 0.53 -> 0.78 purely from fixing labels.

Production config (A) at n=100: groundedness **0.96** [0.95, 0.98], correctness
**0.97** [0.94, 0.99], citation precision 1.00, behaviour-match 0.98,
**$0.0157/query**, **10.1s**. The intervals are now ±0.02 instead of ±0.05.
-> `python -m eval.generate_questions --audit`, `eval/eval_report_n100.md`

**Known gap:** only 2 unanswerable questions exist (from the curated set), so the
refusal claim rests on thin evidence. The generated set is all answerable.

### D-11 · Observability: traces, not just counters
Instrumentation counted cost and latency but discarded it after printing. Traces
are now persisted per request (`traces/traces.jsonl`, append-only JSONL) with a
trace id, per-stage spans, and **chunk attribution**, for any stored answer you
can ask which chunks produced it, at what rank and RRF score.

Retrieval is traced as its own span type. Without it a trace accounts for LLM
time only and silently attributes retrieval latency to nothing, while the
percentages still sum to 100%.

Local JSONL rather than a hosted backend (Langfuse/Phoenix) is deliberate: a
hosted tool adds a UI but makes reproducing this repo depend on someone else's
account. **Honest gap:** there is no trace UI, and no sampling of production
traffic back into the eval set.
-> `python -m src.trace_view --last | --slowest 3 | --doc <file.pdf>`

### D-12 · The judge was checked two ways that need no human, and both have limits
**Cross-judge** (`--cross-judge`): `claude-opus-5` vs `claude-opus-4-8` over 30
answers, 100% raw agreement, κ = 1.0. **This is weaker evidence than it looks.**
Chance agreement is 93.6% because both judges rate almost everything acceptable,
so a single disagreement would move κ by 0.52. The tool now reports that
fragility automatically. And both are Claude models: a shared blind spot yields
perfect agreement while both are wrong. It rules out idiosyncrasy, not error.

**Deterministic key-fact coverage** (`--spot-check`): no LLM at all, so no shared
bias. 92.5% mean coverage, 82/100 with every key fact present, and **zero** cases
of the judge crediting an answer containing none of its key facts.

The first version of that check reported **17 defects out of 100, all false**.
It required every content word of a fact to appear exactly, so an answer saying
"you can use `psql` ... to connect directly" failed a fact worded "psql is the
interactive terminal program used to connect directly" on `use` vs `used`.
Reading the disagreements found the bug in *the check*, not the judge. Now
threshold-based (70% of content words, crude stemming).

**Still missing:** an external, non-Claude judgment. Nothing above closes that.
-> `python -m eval.label_for_kappa --cross-judge --spot-check`

### D-13 · The deployment served nothing, and two things had to be baked in
The previous blueprint told you to upload 223MB of PDFs to a Render disk and
let the app ingest on boot. Until you did, the container came up **healthy and
refused every question**: the synthesizer correctly declines with no evidence,
so a broken deploy looked like a working one.

Two fixes, both verified on the built image rather than assumed:

* **A pre-built index is baked in** (`demo_index/`, 46 documents, 87 chunks,
  3.8MB), chosen deterministically as every document the curated benchmark
  references plus a seeded sample. `DEMO_MODE=true` makes the UI state that it
  serves a subset, because the published metrics were measured on all 733 and
  conflating the two would overstate the demo.
* **The ONNX embedding model is baked in.** ChromaDB lazily downloads it
  (~79MB) on the *first embedding call*, so the first user request paid for a
  network fetch and would fail outright without egress. Confirmed fixed by
  running retrieval with `--network none`.

Measured: image 1.13GB, peak RSS ~282MB per query, health endpoint 200 in ~20s,
one real in-container query = 1 LLM call at $0.0215.
-> `python -m src.build_demo_index && docker build -t citera:demo .`

---

## Corrections made (kept deliberately)

Recorded because the failure modes are more instructive than the fixes.

- **A "verified" README claim the linked artifact contradicted.** The README said
  Q2 was confirmed by recall@k=1.0; `metrics.json` recorded 0.00.
- **A diagnostic that did not mirror production.** It measured retrieval at
  `top_k=50` while the agent ran `top_k=10`, and because RRF is not monotonic it
  produced a confident, wrong conclusion about the bottleneck. A diagnostic that
  does not mirror production is worse than no diagnostic.
- **Multilingual answering silently corrupted the eval.** The refusal marker was
  English-only, so a translated refusal scored as an answer.
- **The harness benchmarked a pipeline nobody runs.** `evaluate()` hard-coded
  `use_planner=True` while production shipped the single-retrieval path. A full
  run silently measured the wrong system: the numbers look fine, they just
  describe something else. Now resolved from config, with a regression test.
- **A stale process kept running after a failed kill.** `pkill` from Git Bash
  cannot see Windows processes; it reported success while the old run continued,
  so two harnesses wrote to the same output concurrently. Verify with
  `Get-CimInstance Win32_Process`, not `pgrep`.
- **Tracing retrieval broke the LLM-call counter.** Retrieval spans landed in
  the same list as LLM spans, so `llm_calls` reported 2 where 1 was correct. The
  metric still looked plausible, which is what made it dangerous. Split by span
  type, with tests.
- **A judge cross-check that flattered itself.** κ = 1.0 on a base rate where
  94% of items share a category: the statistic had almost no room to move. The
  headline number was strong and the evidence behind it was thin.
- **A validation check that was wrong about the thing it validated.** Exact
  word matching flagged 17 correct answers as defective. Every "failure" the
  tool reports has to be read before it is believed.
- **A retriever that mixed two different indexes.** BM25 paths were module
  constants, so a retriever pointed at a fresh directory paired an EMPTY vector
  store with the DEFAULT BM25 index and reported `bm25_ready=True`. Paths now
  derive from the persist directory.
- **An empty index raised an opaque ChromaDB `TypeError`** ("requested results
  0") instead of reporting the operational state it actually was.
- **A UI bug that the ablation would have shipped.** `app/main.py` hand-rolled its
  initial state; with the planner off, every answer would have become a refusal.

---

## Explicitly out of scope

Listed so they are visibly *declined*, not forgotten. Each is defensible as a
"what would you do next" answer without being built.

- **Cross-encoder reranker**: only useful paired with a wider candidate pool
  (D-5), and retrieval is already 1.00 (D-4). No headroom to buy.
- **Contextual Retrieval / semantic chunking**: both target long multi-page
  documents; this corpus is single-page articles.
- **Boilerplate stripping at ingest**: 75% of chunks carry print-to-PDF headers,
  but only ~4-6% of words. Measured, judged not worth the ingest complexity.
- **Multi-lingual answering**: currently a liability, not a feature. Needs a
  language-aware refusal detector before it is worth advertising.
- **Fine-tuning**: no evidence any failure here is a model-capability problem.
- **Multi-turn conversation memory**: a different product, not this one.
