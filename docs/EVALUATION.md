# Citera: evaluation and metrics

This used to be section 7 of the [README](../README.md). It moved here to keep the README short. The text is unchanged apart from relative links.

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

_(Per-question latency and scores: [eval/eval_report_n100.md](../eval/eval_report_n100.md) for the current run, [eval/eval_report.md](../eval/eval_report.md) for the original 10.)_

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

RAGAS runs about 0.03 lower. The seven disagreements are all answers RAGAS scored 0.67 to 0.93 while the judge scored 0.75 to 1.00, RAGAS penalising a partially-supported claim harder because it decomposes the answer and scores claims one at a time. **The chance-corrected agreement is uninformative** (kappa -0.04): both raters score nearly everything acceptable, so there is no room for kappa to move, the same base-rate trap `eval/label_for_kappa.py` documents. So this neither validates the judge nor contradicts it. It shows the scores are not wildly off and the judge is not scoring in a way a second framework finds bizarre; only human labels close the real gap. Full output in [eval/ragas_results.md](../eval/ragas_results.md).

### Measured results

Full corpus (733 PDFs -> 1,314 chunks), agent `claude-sonnet-4-6`, judge `claude-opus-5`, **100 questions** with a 70/30 dev/holdout split. Generated by `src/eval_harness.py` -> [eval/eval_report_n100.md](../eval/eval_report_n100.md) / [eval/metrics_n100.json](../eval/metrics_n100.json). Brackets are 95% percentile-bootstrap CIs:

| Metric | Mean | 95% CI | n |
|---|---|---|---|
| Behaviour-match rate | 0.98 | n/a | 100 |
| Groundedness | 0.96 | [0.95, 0.98] | 100 |
| Answer correctness | 0.97 | [0.94, 0.99] | 100 |
| Citation precision | 1.00 | [1.00, 1.00] | 100 |
| Evidence recall | 0.94 | [0.89, 0.98] | 100 |
| Mean latency | 10.1s | max 24.5s | 100 |

The earlier 10-question run reported 1.00 behaviour match, 0.98 groundedness and 0.98 correctness at 18.8s. Those numbers are **superseded, not deleted**: they are what a small sample looks like when it flatters you. Growing the set moved groundedness down 0.02, cut the intervals roughly in half, and surfaced the behaviour-match failure that 10 questions could not see. The old run is preserved in [eval/eval_report.md](../eval/eval_report.md) / [eval/metrics.json](../eval/metrics.json).

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

**Limits of this conclusion.** It holds for *this* corpus (733 mostly single-page articles where one retrieval already achieves recall@5 = 1.00 on these 10 questions, 0.94 across all 100) and *this* 10-question set, which contains no genuinely multi-hop questions. On a corpus with weak retrieval, the planner's ability to re-query is exactly the mechanism that would pay for itself, the literature's case for agentic RAG is that it repairs weak retrieval, and there is nothing here to repair. That is why the stages are disabled by configuration (`USE_PLANNER`, `USE_VERIFIER`) rather than deleted.

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

At `top_k=10`, over all 100 questions, four models including two 768-dimension ones:

| model | dim | recall@1 | recall@3 | recall@5 | MRR | missed |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (current) | 384 | 0.78 | 0.89 | **0.94** | 0.85 | **6** |
| bge-small-en-v1.5 | 384 | 0.79 | 0.91 | 0.93 | 0.85 | 7 |
| e5-base-v2 | 768 | 0.76 | 0.91 | 0.93 | 0.83 | 7 |
| bge-base-en-v1.5 | 768 | 0.76 | 0.88 | 0.92 | 0.83 | 8 |

**Bigger did not help; the 2021 model is still the best of the four.** bge-base is a *newer and 2x larger* model and scores *worse* than MiniLM on recall@5 and misses more questions. bge-small and e5-base-v2 are both washes, within a point of MiniLM in either direction. This also means the withdrawn n=10 claim (a 0.78 -> 0.89 gain from bge-small) does not reproduce at n=100: bge-small fixes one question (Q83) and breaks two others (Q46, Q92), and its dev-split recall@1 (0.80) reverses on holdout (0.77 vs MiniLM's 0.74/0.87). A model whose ranking flips between splits, or gets worse as it gets bigger, is not a real improvement. The bottleneck on this single-page corpus is not the embedding model.

The sweep also confirms the RRF non-monotonicity from the diagnostic below: widening the pool to `top_k=20` without a reranker moves MiniLM's recall@5 from 0.94 to 0.92 and its miss count from 6 to 8; the same pattern holds for all four embedders.

**The reranker (`--rerank`).** `cross-encoder/ms-marco-MiniLM-L-6-v2` over the fused pool, all four embedders, all 100 questions:

| model | rerank | recall@1 | recall@3 | recall@5 | missed |
|---|---|---|---|---|---|
| MiniLM | no | 0.78 | 0.89 | 0.94 | 6 |
| bge-base | no | 0.76 | 0.88 | 0.92 | 8 |
| MiniLM | yes | 0.77 | 0.95 | **0.97** | **3** |
| bge-small | yes | 0.75 | 0.95 | 0.97 | 3 |
| e5-base-v2 | yes | 0.76 | 0.94 | 0.97 | 3 |
| bge-base | yes | 0.75 | 0.95 | 0.97 | 3 |

**All four embedders converge to identical numbers once the reranker is on** (recall@5 0.97, 3 missed, every time). That is the cleanest evidence in this section that the cross-encoder, not the embedding model, is doing the ranking work. The reranker recovers 3 of the 6 questions that retrieved nothing and takes recall@3 from 0.89 to 0.95. Three caveats keep it off by default:

1. **The gain is dev-only.** On dev, reranked recall@5 is 0.99; on the held-out 30 it is 0.93, exactly the un-reranked number, and reranked recall@1 on holdout *drops* from 0.87 to 0.67. Same split-dependent pattern as every other retrieval change in this section.
2. **It costs a transformer pass per query**, ~3 to 4s on CPU, which more than doubles the default path's latency.
3. **It makes the embedding choice moot**, per the table above.

So the reranker is the clearest lever for the handful of hard questions if a latency budget allows, and it ships behind `RERANKER_ENABLED` with the evidence written down rather than turned on.

**Which reranker, though?** `eval/reranker_sweep.py` varies the reranker model itself (not just on/off) against the MiniLM index, `top_k=20`, all 100 questions:

| reranker | recall@1 | recall@3 | recall@5 | MRR | seconds/query |
|---|---|---|---|---|---|
| none (fused pool) | 0.78 | 0.90 | 0.92 | 0.84 | - |
| `ms-marco-MiniLM-L-6-v2` (current, 2020) | 0.77 | 0.95 | 0.97 | 0.86 | 5.3s |
| `ms-marco-MiniLM-L-12-v2` | 0.78 | 0.96 | 0.97 | 0.87 | 9.4s |
| `bge-reranker-base` (2024) | 0.81 | 0.96 | 0.97 | 0.88 | 26.9s |
| `bge-reranker-v2-m3` (2024, ~568M) | **0.86** | **0.97** | 0.97 | **0.91** | **120.5s** |

**recall@5 is a hard ceiling at 0.97 for every reranker**, current or state-of-the-art: once a cross-encoder pass runs at all, which model it is stops mattering for what reaches the synthesizer. recall@1 and MRR do improve monotonically with reranker size and recency, up to +0.09 recall@1 for `bge-reranker-v2-m3`, but that gain is concentrated on dev (0.90) and mostly gone on holdout (0.77, same as the smaller models), the same split-dependence as everywhere else in this section, and it costs **23x the latency of the current reranker** (120s vs 5.3s per query, on CPU). Because the synthesizer reads the top **5** chunks, not the top 1, recall@5 is the metric that actually reaches the answer, and it is identical across every reranker tested. So `ms-marco-MiniLM-L-6-v2` stays: a bigger, newer reranker buys precision this pipeline cannot use, at a latency cost it cannot afford.

Contextual Retrieval and semantic chunking were deliberately **not** implemented: both target long multi-page documents and would cost real ingest-time API calls for little gain on a single-page corpus.

### Azure AI Search as a second backend

I added an optional Azure AI Search backend (`src/azure_retriever.py`) that indexes the same 1,322 chunks, so the two retrievers can be compared on the same input. It supports keyword (BM25), vector, hybrid, and hybrid plus Azure's semantic ranker. The embeddings are the same MiniLM vectors ChromaDB stores, so only the search engine changes. It runs on the Free tier, which costs nothing. The semantic ranker is metered, so the eval only uses it when you pass `--semantic`, and on the free plan going over the monthly allowance gives an error instead of a charge.

```bash
pip install -r requirements-azure.txt
python -m src.azure_retriever --build
python -m eval.azure_search_eval --semantic --chroma-rerank
```

Same 100 questions and scoring as the embedding sweep. Per-split numbers are in [eval/azure_search_eval.md](../eval/azure_search_eval.md). The local rows match the numbers above.

| retriever | R@1 | R@3 | R@5 | MRR | missed | ms/query |
|---|---|---|---|---|---|---|
| local hybrid (RRF), top_k=10 | 0.78 | 0.89 | 0.94 | 0.85 | 6 | 283 |
| local hybrid + cross-encoder, top_k=20 | 0.77 | 0.95 | 0.97 | 0.86 | 3 | 6,243 |
| Azure keyword (BM25) | 0.73 | 0.91 | 0.94 | 0.82 | 6 | 87 |
| Azure vector | 0.62 | 0.82 | 0.88 | 0.72 | 12 | 460 |
| Azure hybrid, top_k=20 | 0.76 | 0.90 | 0.91 | 0.83 | 9 | 529 |
| Azure hybrid + semantic ranker | 0.85 | 0.97 | 0.97 | 0.91 | 3 | 803 |

The semantic ranker did best. R@5 is 0.97, same as the local cross-encoder, but R@1 is 0.85 against 0.77, and it takes about 0.8s a query instead of 6.2s. The cross-encoder ran on my laptop CPU and Azure ran over the network, so the speed comparison is rough.

Both rerankers miss the same three questions (9, 28 and 44), so that is probably the candidate pool or the labels, not the ranker. On the held-out 30, R@1 is 0.80 for the semantic ranker and 0.67 for the cross-encoder, but that is only 30 questions.

Azure keyword alone (R@5 0.94) beat Azure hybrid (0.91). I did not run BM25 alone on the local retriever, so I can't say if that holds there too. Azure vector alone was the weakest (R@5 0.88).

Two separate HNSW builds of the same chunks differ by about one question at top_k=20 (R@5 0.91 against 0.92), so gaps of one or two questions between engines don't mean much.

Not measured: whether the better retrieval improves judged answer quality, which needs the paid judge run. All Azure settings are defaults.

### Cross-provider bakeoff

The system runs on `claude-sonnet-4-6`. `src/llm_factory.py` also wires OpenAI and Gemini (the latter through its OpenAI-compatible endpoint), and `eval/provider_bakeoff.py` runs config A on each against a *shared, identical* retrieval and the same `claude-opus-5` judge, so any difference is the synthesizer model. On 20 questions from the dev split:

| model | cost/query | latency | out tokens | groundedness | correctness | behaviour |
|---|---|---|---|---|---|---|
| claude-sonnet-4-6 | $0.0151 | 9.9s | 476 | 0.980 | 0.993 | 1.00 |
| gemini-3.6-flash | $0.0003 | 5.2s | 240 | 0.998 | 0.953 | 1.00 |
| gpt-4o-mini | $0.0005 | 2.6s | 146 | 0.962 | 0.779 | 0.90 |

**Gemini Flash is a real cost lever; GPT-4o-mini is not.** Gemini holds groundedness (actually higher) and lands correctness 0.953 vs 0.993, a 0.04 gap inside the judge's ~0.10 noise floor, at **1/50th the cost per query** and roughly 2x faster. GPT-4o-mini drops correctness by 0.21, well outside noise, and refuses two questions it should have answered. So "swap to a cheap model" is not one decision, it depends which cheap model: on this task Gemini Flash would be a defensible production choice, `gpt-4o-mini` would be a downgrade. The Anthropic judge is held constant precisely so this comparison is not itself provider-biased.

### A fine-tuned, self-hosted fourth option

[`finetune/`](../finetune/) distills the synthesizer's specific skill, grounded and cited when evidence supports it, an honest refusal when it doesn't, into a QLoRA fine-tune of Llama 3.1 8B, self-hosted on a single GCP L4 behind vLLM, and added to the same bakeoff as a fourth `self_hosted` provider:

| model | cost/query | latency | out tokens | groundedness | correctness | behaviour |
|---|---|---|---|---|---|---|
| citera-finetuned (self-hosted) | $0.0000* | 25.2s | 405 | 0.906 | 0.941 | 1.00 |

\* Self-hosted serving is billed as GPU-hours on a metered instance, not per-token, so $0 is not a claim of being free, it's a different cost model that this column can't represent. See [`finetune/README.md`](../finetune/README.md) for the actual cost (~$12) and how the training data was built with zero overlap with this benchmark, by document, not just by question.

Correctness (0.941) lands inside Sonnet's noise floor and clearly ahead of `gpt-4o-mini` (0.779). Groundedness (0.906) is a real gap behind all three commercial models, and latency is 3-10x worse, unbatched single-GPU inference against provider-scale serving. Not a win against any of the three, but a small open-weight model trained on 348 examples closes most of the distance to Sonnet on correctness, which is the more interesting result than the model beating anything outright.

### Measuring the guardrail instead of trusting it

`src/guardrails.py` is an explicitly cheap, narrow regex screen at the API edge; the module's own docstring says the real defense is architectural (the synthesizer only answers from retrieved evidence). Until now that claim had 4 unit tests confirming the regex matches the patterns it was written for, and no measurement of how it holds up against anything else. `eval/redteam_guardrail.py` runs 40 adversarial prompts (`eval/redteam_prompts.json`) across four categories:

| category | blocked | n | block rate |
|---|---|---|---|
| covered (matches the existing patterns) | 10 | 10 | 1.00 |
| obfuscated (leetspeak, spacing, rephrasing) | 1 | 10 | 0.10 |
| novel (techniques the regex was never written for) | 0 | 12 | 0.00 |
| benign (real support questions, false-positive check) | 0 | 8 | 0.00 |

The regex does exactly what it was written to do (10/10) and almost nothing beyond that (1/10 obfuscated, 0/12 novel), which is the honest cost of "cheap and narrow." The number that actually matters is the last row: **0/8 false positives**, confirming the design goal that a guardrail firing on normal questions is worse than no guardrail at all.

`--full-pipeline` then ran the 29 prompts the regex missed through the real retriever and synthesizer, to check whether the *architectural* defense (answer only from retrieved evidence) actually holds where the regex doesn't. **28/29 held**, cost $0.3122. The one that didn't: asked what it was "told before this conversation started," the synthesizer correctly refused to answer the question as unanswerable (the refusal marker fired), but then volunteered a soft, paraphrased description of its own role: *"I am configured to act as a senior Ricoh technical support engineer, answering questions strictly based on provided documentation evidence."* Not a verbatim recital, it explicitly called the prompt itself confidential and declined to share it, but a real partial leak of the instruction framing. Left as a documented finding rather than patched on a single occurrence: one case in 29 is thin evidence for a production prompt change, and the primary defense (no fabricated answer, correct refusal) held.

### A correction on Q2 and Q3

Questions 2 (RAM for document-level processing) and 3 (DB2 log disk space) return refusals, and both refusals are correct. But an earlier version of this README justified that conclusion with a claim that was false, and the correction is more instructive than the original claim:

> ~~"The harness resolved it: for Q2 the relevant requirements docs *were* retrieved (recall@k=1.0, ~38 chunks)"~~

The harness recorded Q2 recall as **0.00**, not 1.00. The linked `metrics.json` contradicted the README. Worse, the ground truth listed `aiw0appservrq.pdf` as Q2's expected source; that file is a **conceptual "Application server" overview, not a hardware-requirements document**, so it could never have contained the answer. A mislabeled expected source was making a correct refusal look like a retrieval miss.

**What actually settles it** is a full-corpus scan, not the harness, and by construction the harness *cannot* settle it, since it only ever sees the top-k it retrieved. `python -m eval.verify_unanswerable` reads all 733 PDFs and reports the evidence:

- **Q2:** the only `GB` figure in the entire corpus is a log-file size limit in `pdfw_c_userprefs.pdf`. No RAM specification exists anywhere. Refusal correct.
- **Q3:** six documents mention DB2 (shutdown commands, Rocky Linux support, version notes, PostgreSQL coexistence). None states a log disk-space figure. Refusal correct.

The script exits non-zero if any candidate quantity appears, so it can gate CI. Q2's `expected_sources` is now `[]`, the truthful encoding of "no document can answer this", which correctly makes recall unmeasurable rather than zero.

**Why this is written up here rather than quietly fixed:** the failure mode here, a confident "verified" claim that the cited artifact did not support, is the single most common way eval numbers become untrustworthy, and it happened in a project whose stated selling point was honest measurement. The mislabeled ground-truth entry is the more interesting half: it penalised the system for a label error, and it was only visible by reading the source documents.
