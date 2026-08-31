# Citera: Quality Evaluation Report

**Generated:** 2026-08-30 20:54:05  
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
| Mean latency | 15.92s | n/a | Per-question wall-clock (max 32.36s) |

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

- **Mean cost per query: $0.02755** (max $0.05060)
- Mean 2.0 LLM calls, 5,828 in / 671 out tokens
- Whole-benchmark agent cost $0.8266; judge overhead $0.0000

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 30 | 377.87s | 79.1% | $0.76402 | 92.4% |
| planner | 30 | 89.06s | 18.7% | $0.06263 | 7.6% |
| retrieval | 171 | 10.52s | 2.2% | $0.00000 | 0.0% |

### Worst case (what the means hide)

- **evidence_recall** = 0.00, Q9: What operating systems does RICOH ProcessDirector run on, and how do users access it?

## Per-question results

| # | Behaviour | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 12.53s | - |
| 2 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 11.7s | - |
| 3 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 17.85s | - |
| 4 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 14.44s | - |
| 5 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 16.78s | - |
| 6 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 21.72s | - |
| 7 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 18.84s | - |
| 8 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 10.68s | - |
| 9 |  | 0.00 | 0.00 | 1.00 | n/a | n/a | 16.56s | retrieval miss |
| 10 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 12.84s | - |
| 11 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 14.22s | - |
| 12 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 12.19s | - |
| 13 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 12.29s | - |
| 14 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 10.48s | - |
| 15 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 12.99s | - |
| 16 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 15.77s | - |
| 17 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 12.23s | - |
| 18 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 16.72s | - |
| 19 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 18.72s | - |
| 20 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 14.9s | - |
| 21 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 14.89s | - |
| 22 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 15.92s | - |
| 23 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 18.24s | - |
| 24 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 13.93s | - |
| 25 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 15.13s | - |
| 26 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 16.0s | - |
| 27 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 12.13s | - |
| 28 |  | 0.00 | 0.00 | 1.00 | n/a | n/a | 23.4s | retrieval miss |
| 29 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 32.36s | - |
| 30 |  | 1.00 | 1.00 | 1.00 | n/a | n/a | 21.27s | - |