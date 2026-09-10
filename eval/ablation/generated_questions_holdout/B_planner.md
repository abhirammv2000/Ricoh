# Citera: Quality Evaluation Report

**Generated:** 2026-09-10 16:35:13  
**Agent model:** `claude-sonnet-4-6`  
**Judge model:** `claude-opus-5`  
**Questions:** 30  

> Every mean carries a 95% percentile-bootstrap confidence interval.
> At this sample size the intervals are wide by construction, they
> are reported so the numbers are not read as more precise than the
> eval set can support.

## Aggregate metrics

| Metric | Mean | 95% CI | What it means |
|---|---|---|---|
| Behaviour-match rate | 1.00 | n/a | Answered vs. refused as expected |
| Evidence recall | 0.93 | [0.83, 1.00] | Expected doc reached the synthesizer, across ALL passes/retries (n=30) |
| Citation precision | 1.00 | [1.00, 1.00] | Cited docs exist in the evidence (catches fabricated filenames only) (n=30) |
| Groundedness | 0.97 | [0.96, 0.98] | Claims supported by evidence (n=30) |
| Correctness | 0.98 | [0.96, 1.00] | Conveys expected facts / refuses correctly (n=30) |
| Mean latency | 18.68s | n/a | Per-question wall-clock (max 31.66s) |

### Retriever in isolation

Single retrieval on the raw question, no planner, no sub-queries,
no entity boost, no retry. Ranks counted over distinct documents.
Comparing this against *Evidence recall* separates a retrieval
failure from a planning failure.

| Depth | Recall |
|---|---|
| recall@1 | 0.87 |
| recall@3 | 0.93 |
| recall@5 | 0.93 |

### Cost and where the time goes

Prices are a dated snapshot (2026-08-01); token counts are
the ground truth and cost is derived from them. Agent cost is what serving
a query costs; judge cost is eval overhead and is never folded into it.

- **Mean cost per query: $0.02675** (max $0.04847)
- Mean 2.0 LLM calls, 5,567 in / 670 out tokens
- Whole-benchmark agent cost $0.8026; judge overhead $1.3941

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 30 | 414.34s | 74.0% | $0.74166 | 92.4% |
| planner | 30 | 122.63s | 21.9% | $0.06095 | 7.6% |
| retrieval | 165 | 23.15s | 4.1% | $0.00000 | 0.0% |
| citation_guardrail | 30 | 0.0s | 0.0% | $0.00000 | 0.0% |

### Worst case (what the means hide)

- **evidence_recall** = 0.00, Q9: What operating systems does RICOH ProcessDirector run on, and how do users access it?
- **groundedness** = 0.88, Q25: When setting up inserter control file rules, where should I copy the sample rules files to, and why is that location recommended?
- **correctness** = 0.75, Q9: What operating systems does RICOH ProcessDirector run on, and how do users access it?

## Per-question results

| # | Behaviour | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 12.16s | - |
| 2 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 18.74s | - |
| 3 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 15.17s | - |
| 4 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 17.34s | - |
| 5 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 18.22s | - |
| 6 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 21.34s | - |
| 7 |  | 1.00 | 1.00 | 1.00 | 0.95 | 0.80 | 19.58s | - |
| 8 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 12.87s | - |
| 9 |  | 0.00 | 0.00 | 1.00 | 0.90 | 0.75 | 19.75s | retrieval miss |
| 10 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.31s | - |
| 11 |  | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 16.13s | - |
| 12 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.06s | - |
| 13 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 21.02s | - |
| 14 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 19.46s | - |
| 15 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 12.3s | - |
| 16 |  | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 15.98s | - |
| 17 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 10.44s | - |
| 18 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 23.31s | - |
| 19 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 22.91s | - |
| 20 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 17.68s | - |
| 21 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 15.66s | - |
| 22 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 20.25s | - |
| 23 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 29.6s | - |
| 24 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 23.4s | - |
| 25 |  | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | 23.75s | - |
| 26 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 13.42s | - |
| 27 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 12.94s | - |
| 28 |  | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 24.06s | retrieval miss |
| 29 |  | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 31.66s | - |
| 30 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 26.94s | - |