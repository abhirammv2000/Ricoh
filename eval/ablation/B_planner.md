# 📊 Citera: Quality Evaluation Report

**Generated:** 2026-08-01 15:14:14  
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
| Behaviour-match rate | 1.00 | — | Answered vs. refused as expected |
| Evidence recall | 0.88 | [0.69, 1.00] | Expected doc reached the synthesizer, across ALL passes/retries (n=8) |
| Citation precision | 1.00 | [1.00, 1.00] | Cited docs exist in the evidence (catches fabricated filenames only) (n=8) |
| Groundedness | 0.99 | [0.97, 1.00] | Claims supported by evidence (n=10) |
| Correctness | 0.97 | [0.93, 1.00] | Conveys expected facts / refuses correctly (n=10) |
| Mean latency | 13.92s | — | Per-question wall-clock (max 20.08s) |

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

- **Mean cost per query: $0.02071** (max $0.03506)
- Mean 2.0 LLM calls, 4,475 in / 486 out tokens
- Whole-benchmark agent cost $0.2071; judge overhead $0.3685

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 10 | 102.23s | 78.5% | $0.19420 | 93.8% |
| planner | 10 | 27.92s | 21.5% | $0.01288 | 6.2% |

### Worst case (what the means hide)

- **evidence_recall** = 0.50, Q1: What property do I set if I want the printers to enable after a restart?
- **groundedness** = 0.93, Q10: What inserters does RPD support?
- **correctness** = 0.85, Q7: What programs does RPD integrate with?

## Per-question results

| # | Behaviour ✓ | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 | ✅ | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 7.17s | — |
| 2 | ✅ | n/a | n/a | n/a | 1.00 | 1.00 | 11.69s | — |
| 3 | ✅ | n/a | n/a | n/a | 1.00 | 1.00 | 6.67s | — |
| 4 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 15.58s | — |
| 5 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.06s | — |
| 6 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 19.84s | — |
| 7 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 0.85 | 20.08s | — |
| 8 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 15.8s | — |
| 9 | ✅ | 0.50 | 1.00 | 1.00 | 1.00 | 0.85 | 16.19s | — |
| 10 | ✅ | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 15.1s | — |