# Citera: Quality Evaluation Report

**Generated:** 2026-08-30 20:46:05  
**Agent model:** `claude-sonnet-4-6`  
**Judge model:** `disabled`  
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
| Mean latency | 11.32s | n/a | Per-question wall-clock (max 23.56s) |

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

- **Mean cost per query: $0.01580** (max $0.02544)
- Mean 1.0 LLM calls, 2,832 in / 487 out tokens
- Whole-benchmark agent cost $0.4739; judge overhead $0.0000

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 30 | 335.14s | 99.3% | $0.47389 | 100.0% |
| retrieval | 30 | 2.26s | 0.7% | $0.00000 | 0.0% |

### Worst case (what the means hide)

- **evidence_recall** = 0.00, Q9: What operating systems does RICOH ProcessDirector run on, and how do users access it?

## Per-question results

| # | Behaviour | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 14.37s | - |
| 2 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 7.38s | - |
| 3 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 10.29s | - |
| 4 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 10.19s | - |
| 5 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 12.26s | - |
| 6 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 9.38s | - |
| 7 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 12.94s | - |
| 8 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 5.24s | - |
| 9 |  | 0.00 | 0.00 | 1.00 | n/a | n/a | 22.46s | retrieval miss |
| 10 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 10.62s | - |
| 11 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 9.84s | - |
| 12 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 12.75s | - |
| 13 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 9.27s | - |
| 14 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 7.22s | - |
| 15 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 8.65s | - |
| 16 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 11.04s | - |
| 17 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 8.37s | - |
| 18 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 10.56s | - |
| 19 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 13.42s | - |
| 20 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 8.0s | - |
| 21 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 11.43s | - |
| 22 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 8.76s | - |
| 23 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 11.52s | - |
| 24 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 8.35s | - |
| 25 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 12.61s | - |
| 26 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 11.01s | - |
| 27 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 8.08s | - |
| 28 |  | 0.00 | 0.00 | 1.00 | n/a | n/a | 11.25s | retrieval miss |
| 29 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 23.56s | - |
| 30 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 18.91s | - |