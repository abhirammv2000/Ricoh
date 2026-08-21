# 📊 Citera: Quality Evaluation Report

**Generated:** 2026-08-01 15:07:24  
**Agent model:** `claude-sonnet-4-6`  
**Judge model:** `claude-opus-5`  
**Questions:** 10  

> Every mean carries a 95% percentile-bootstrap confidence interval.
> At this sample size the intervals are wide by construction, they
> are reported so the numbers are not read as more precise than the
> eval set can support.

## Aggregate metrics

| Metric | Mean | 95% CI | What it means |
|---|---|---|---|
| Behaviour-match rate | 1.00 | n/a | Answered vs. refused as expected |
| Evidence recall | 0.88 | [0.69, 1.00] | Expected doc reached the synthesizer, across ALL passes/retries (n=8) |
| Citation precision | 1.00 | [1.00, 1.00] | Cited docs exist in the evidence (catches fabricated filenames only) (n=8) |
| Groundedness | 0.97 | [0.95, 0.99] | Claims supported by evidence (n=10) |
| Correctness | 0.98 | [0.94, 1.00] | Conveys expected facts / refuses correctly (n=10) |
| Mean latency | 15.77s | n/a | Per-question wall-clock (max 23.68s) |

### Retriever in isolation

Single retrieval on the raw question, no planner, no sub-queries,
no entity boost, no retry. Ranks counted over distinct documents.
Comparing this against *Evidence recall* separates a retrieval
failure from a planning failure.

| Depth | Recall |
|---|---|
| recall@1 | 0.44 |
| recall@3 | 0.94 |
| recall@5 | 1.00 |

### Cost and where the time goes

Prices are a dated snapshot (2026-08-01); token counts are
the ground truth and cost is derived from them. Agent cost is what serving
a query costs; judge cost is eval overhead and is never folded into it.

- **Mean cost per query: $0.05155** (max $0.14623)
- Mean 3.4 LLM calls, 14,448 in / 547 out tokens
- Whole-benchmark agent cost $0.5155; judge overhead $0.5274

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 10 | 102.74s | 67.7% | $0.26544 | 51.5% |
| planner | 12 | 30.28s | 20.0% | $0.01900 | 3.7% |
| verifier | 12 | 18.72s | 12.3% | $0.23108 | 44.8% |

### Worst case (what the means hide)

- **evidence_recall** = 0.50, Q1: What property do I set if I want the printers to enable after a restart?
- **groundedness** = 0.90, Q1: What property do I set if I want the printers to enable after a restart?
- **correctness** = 0.80, Q9: How do I use locations?

## Per-question results

| # | Behaviour ✓ | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 | ✅ | 0.50 | 1.00 | 1.00 | 0.90 | 1.00 | 11.67s | n/a |
| 2 | ✅ | n/a | n/a | n/a | 1.00 | 1.00 | 23.68s | n/a |
| 3 | ✅ | n/a | n/a | n/a | 1.00 | 1.00 | 15.2s | n/a |
| 4 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 18.74s | n/a |
| 5 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.99s | n/a |
| 6 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 19.35s | n/a |
| 7 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 21.44s | n/a |
| 8 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 15.55s | n/a |
| 9 | ✅ | 0.50 | 1.00 | 1.00 | 0.95 | 0.80 | 9.95s | n/a |
| 10 | ✅ | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 9.11s | n/a |