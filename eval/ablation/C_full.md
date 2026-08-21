# Citera quality evaluation report

**Generated:** 2026-08-01 15:17:31  
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
| Groundedness | 0.98 | [0.95, 0.99] | Claims supported by evidence (n=10) |
| Correctness | 0.98 | [0.94, 1.00] | Conveys expected facts / refuses correctly (n=10) |
| Mean latency | 15.61s | n/a | Per-question wall-clock (max 21.58s) |

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

- **Mean cost per query: $0.04902** (max $0.13534)
- Mean 3.4 LLM calls, 13,609 in / 546 out tokens
- Whole-benchmark agent cost $0.4902; judge overhead $0.4901

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 10 | 103.53s | 68.5% | $0.25310 | 51.6% |
| planner | 12 | 24.62s | 16.3% | $0.01858 | 3.8% |
| verifier | 12 | 22.97s | 15.2% | $0.21848 | 44.6% |

### Worst case (what the means hide)

- **evidence_recall** = 0.50, Q1: What property do I set if I want the printers to enable after a restart?
- **groundedness** = 0.90, Q1: What property do I set if I want the printers to enable after a restart?
- **correctness** = 0.80, Q9: How do I use locations?

## Per-question results

| # | Behaviour | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 | yes | 0.50 | 1.00 | 1.00 | 0.90 | 1.00 | 14.43s | n/a |
| 2 | yes | n/a | n/a | n/a | 1.00 | 1.00 | 17.01s | n/a |
| 3 | yes | n/a | n/a | n/a | 1.00 | 1.00 | 12.49s | n/a |
| 4 | yes | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 18.11s | n/a |
| 5 | yes | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.64s | n/a |
| 6 | yes | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 21.58s | n/a |
| 7 | yes | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 19.44s | n/a |
| 8 | yes | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 15.39s | n/a |
| 9 | yes | 0.50 | 1.00 | 1.00 | 0.95 | 0.80 | 14.77s | n/a |
| 10 | yes | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.21s | n/a |