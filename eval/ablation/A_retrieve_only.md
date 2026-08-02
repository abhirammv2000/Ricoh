# 📊 Citera — Quality Evaluation Report

**Generated:** 2026-08-01 15:11:00  
**Agent model:** `claude-sonnet-4-6`  
**Judge model:** `claude-opus-5`  
**Questions:** 10  

> Every mean carries a 95% percentile-bootstrap confidence interval.
> At this sample size the intervals are wide by construction — they
> are reported so the numbers are not read as more precise than the
> eval set can support.

## Aggregate metrics

| Metric | Mean | 95% CI | What it means |
|---|---|---|---|
| Behaviour-match rate | 1.00 | — | Answered vs. refused as expected |
| Evidence recall | 1.00 | [1.00, 1.00] | Expected doc reached the synthesizer, across ALL passes/retries (n=8) |
| Citation precision | 1.00 | [1.00, 1.00] | Cited docs exist in the evidence (catches fabricated filenames only) (n=8) |
| Groundedness | 0.98 | [0.97, 1.00] | Claims supported by evidence (n=10) |
| Correctness | 1.00 | [1.00, 1.00] | Conveys expected facts / refuses correctly (n=10) |
| Mean latency | 9.89s | — | Per-question wall-clock (max 16.59s) |

### Retriever in isolation

Single retrieval on the raw question — no planner, no sub-queries,
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

- **Mean cost per query: $0.01593** (max $0.02165)
- Mean 1.0 LLM calls, 2,850 in / 492 out tokens
- Whole-benchmark agent cost $0.1593; judge overhead $0.2664

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 10 | 95.7s | 100.0% | $0.15931 | 100.0% |

### Worst case (what the means hide)

- **evidence_recall** = 1.00 — Q1: What property do I set if I want the printers to enable after a restart?
- **groundedness** = 0.95 — Q1: What property do I set if I want the printers to enable after a restart?
- **correctness** = 1.00 — Q1: What property do I set if I want the printers to enable after a restart?

## Per-question results

| # | Behaviour ✓ | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 7.47s | — |
| 2 | ✅ | n/a | n/a | n/a | 1.00 | 1.00 | 3.81s | — |
| 3 | ✅ | n/a | n/a | n/a | 1.00 | 1.00 | 5.43s | — |
| 4 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.82s | — |
| 5 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.18s | — |
| 6 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 16.59s | — |
| 7 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.19s | — |
| 8 | ✅ | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 13.16s | — |
| 9 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.92s | — |
| 10 | ✅ | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 6.34s | — |