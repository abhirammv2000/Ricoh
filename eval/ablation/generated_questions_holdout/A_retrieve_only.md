# Citera: Quality Evaluation Report

**Generated:** 2026-09-10 16:23:29  
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
| Groundedness | 0.96 | [0.94, 0.98] | Claims supported by evidence (n=30) |
| Correctness | 0.97 | [0.92, 1.00] | Conveys expected facts / refuses correctly (n=30) |
| Mean latency | 12.09s | n/a | Per-question wall-clock (max 21.69s) |

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

- **Mean cost per query: $0.01606** (max $0.02475)
- Mean 1.0 LLM calls, 2,832 in / 505 out tokens
- Whole-benchmark agent cost $0.4819; judge overhead $0.8378

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 30 | 359.73s | 99.6% | $0.48194 | 100.0% |
| retrieval | 30 | 1.59s | 0.4% | $0.00000 | 0.0% |
| citation_guardrail | 30 | 0.0s | 0.0% | $0.00000 | 0.0% |

### Worst case (what the means hide)

- **evidence_recall** = 0.00, Q9: What operating systems does RICOH ProcessDirector run on, and how do users access it?
- **groundedness** = 0.80, Q14: Which specific RICOH Pro 8400 series printer models are supported in ProcessDirector, and what digital front end do they require?
- **correctness** = 0.35, Q9: What operating systems does RICOH ProcessDirector run on, and how do users access it?

## Per-question results

| # | Behaviour | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.53s | - |
| 2 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.01s | - |
| 3 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.93s | - |
| 4 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.01s | - |
| 5 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.18s | - |
| 6 |  | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 11.52s | - |
| 7 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 16.1s | - |
| 8 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.06s | - |
| 9 |  | 0.00 | 0.00 | 1.00 | 0.82 | 0.35 | 13.93s | retrieval miss |
| 10 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 13.85s | - |
| 11 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.57s | - |
| 12 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.62s | - |
| 13 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 12.26s | - |
| 14 |  | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 | 10.59s | - |
| 15 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.22s | - |
| 16 |  | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 10.43s | - |
| 17 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.85 | 8.55s | - |
| 18 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.46s | - |
| 19 |  | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | 16.46s | - |
| 20 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.94s | - |
| 21 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.47s | - |
| 22 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 13.68s | - |
| 23 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.65s | - |
| 24 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.36s | - |
| 25 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 13.37s | - |
| 26 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 11.88s | - |
| 27 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 12.28s | - |
| 28 |  | 0.00 | 0.00 | 1.00 | 0.90 | 0.85 | 8.03s | retrieval miss |
| 29 |  | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | 21.69s | - |
| 30 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 18.15s | - |