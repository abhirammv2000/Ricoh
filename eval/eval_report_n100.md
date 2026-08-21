# 📊 Citera: Quality Evaluation Report

**Generated:** 2026-08-01 17:22:48  
**Agent model:** `claude-sonnet-4-6`  
**Judge model:** `claude-opus-5`  
**Questions:** 100  

> Every mean carries a 95% percentile-bootstrap confidence interval.
> At this sample size the intervals are wide by construction, they
> are reported so the numbers are not read as more precise than the
> eval set can support.

## Aggregate metrics

| Metric | Mean | 95% CI | What it means |
|---|---|---|---|
| Behaviour-match rate | 0.98 | n/a | Answered vs. refused as expected |
| Evidence recall | 0.94 | [0.89, 0.98] | Expected doc reached the synthesizer, across ALL passes/retries (n=100) |
| Citation precision | 1.00 | [1.00, 1.00] | Cited docs exist in the evidence (catches fabricated filenames only) (n=100) |
| Groundedness | 0.96 | [0.95, 0.98] | Claims supported by evidence (n=100) |
| Correctness | 0.97 | [0.94, 0.99] | Conveys expected facts / refuses correctly (n=100) |
| Mean latency | 10.14s | n/a | Per-question wall-clock (max 24.54s) |

### Retriever in isolation

Single retrieval on the raw question, no planner, no sub-queries,
no entity boost, no retry. Ranks counted over distinct documents.
Comparing this against *Evidence recall* separates a retrieval
failure from a planning failure.

| Depth | Recall |
|---|---|
| recall@1 | 0.78 |
| recall@3 | 0.89 |
| recall@5 | 0.94 |

### Cost and where the time goes

Prices are a dated snapshot (2026-08-01); token counts are
the ground truth and cost is derived from them. Agent cost is what serving
a query costs; judge cost is eval overhead and is never folded into it.

- **Mean cost per query: $0.01571** (max $0.02670)
- Mean 1.0 LLM calls, 2,781 in / 491 out tokens
- Whole-benchmark agent cost $1.5714; judge overhead $2.7179

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 100 | 1000.32s | 100.0% | $1.57144 | 100.0% |

### Worst case (what the means hide)

- **evidence_recall** = 0.00, Q9: What operating systems does RICOH ProcessDirector run on, and how do users access it?
- **groundedness** = 0.55, Q83: If a job misses its first deadline but then finishes within the second deadline, does the red dot stay in the Deadlines portlet or get cleared?
- **correctness** = 0.15, Q83: If a job misses its first deadline but then finishes within the second deadline, does the red dot stay in the Deadlines portlet or get cleared?

## Per-question results

| # | Behaviour ✓ | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.09s | n/a |
| 2 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 9.01s | n/a |
| 3 | ✅ | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 10.43s | n/a |
| 4 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.22s | n/a |
| 5 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.29s | n/a |
| 6 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 10.93s | n/a |
| 7 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.52s | n/a |
| 8 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 5.0s | n/a |
| 9 | ✅ | 0.00 | 0.00 | 1.00 | 0.85 | 0.35 | 13.05s | ✗ retrieval miss |
| 10 | ✅ | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 7.6s | n/a |
| 11 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.63s | n/a |
| 12 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.95s | n/a |
| 13 | ✅ | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 10.28s | n/a |
| 14 | ✅ | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 6.92s | n/a |
| 15 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.5s | n/a |
| 16 | ✅ | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 10.36s | n/a |
| 17 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 0.85 | 8.68s | n/a |
| 18 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.75s | n/a |
| 19 | ✅ | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 13.5s | n/a |
| 20 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.28s | n/a |
| 21 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 9.37s | n/a |
| 22 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.13s | n/a |
| 23 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.97s | n/a |
| 24 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.75s | n/a |
| 25 | ✅ | 1.00 | 1.00 | 1.00 | 0.90 | 1.00 | 10.15s | n/a |
| 26 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 10.83s | n/a |
| 27 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.97s | n/a |
| 28 | ✅ | 0.00 | 0.00 | 1.00 | 1.00 | 0.85 | 7.98s | ✗ retrieval miss |
| 29 | ✅ | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 24.54s | n/a |
| 30 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 17.27s | n/a |
| 31 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.44s | n/a |
| 32 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 6.73s | n/a |
| 33 | ✅ | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 7.36s | n/a |
| 34 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 10.56s | n/a |
| 35 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.21s | n/a |
| 36 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.11s | n/a |
| 37 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.67s | n/a |
| 38 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 9.35s | n/a |
| 39 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 6.13s | n/a |
| 40 | ✅ | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 5.38s | n/a |
| 41 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.52s | n/a |
| 42 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.85s | n/a |
| 43 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 10.98s | n/a |
| 44 | ✅ | 0.00 | 0.00 | 1.00 | 0.80 | 0.80 | 6.15s | ✗ retrieval miss |
| 45 | ✅ | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 8.48s | n/a |
| 46 | ✅ | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | 8.11s | n/a |
| 47 | ✅ | 1.00 | 1.00 | 1.00 | 0.85 | 1.00 | 11.48s | n/a |
| 48 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 15.47s | n/a |
| 49 | ✅ | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 14.52s | n/a |
| 50 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.33s | n/a |
| 51 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.4s | n/a |
| 52 | ✅ | 1.00 | 1.00 | 1.00 | 0.92 | 0.95 | 12.66s | n/a |
| 53 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 14.05s | n/a |
| 54 | ✅ | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 16.76s | n/a |
| 55 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.48s | n/a |
| 56 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 6.56s | n/a |
| 57 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.8s | n/a |
| 58 | ❌ | 0.00 | 0.00 | 1.00 | 0.95 | 0.50 | 10.76s | ✗ behaviour, ✗ retrieval miss |
| 59 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.27s | n/a |
| 60 | ✅ | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 13.58s | n/a |
| 61 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.01s | n/a |
| 62 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.42s | n/a |
| 63 | ✅ | 1.00 | 1.00 | 1.00 | 0.97 | 0.93 | 11.71s | n/a |
| 64 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.73s | n/a |
| 65 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 8.38s | n/a |
| 66 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.34s | n/a |
| 67 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.61s | n/a |
| 68 | ✅ | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 14.62s | n/a |
| 69 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 16.8s | n/a |
| 70 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.81s | n/a |
| 71 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.53s | n/a |
| 72 | ✅ | 0.00 | 0.00 | 1.00 | 0.97 | 0.85 | 14.5s | ✗ retrieval miss |
| 73 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.23s | n/a |
| 74 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.88s | n/a |
| 75 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.47s | n/a |
| 76 | ✅ | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 8.26s | n/a |
| 77 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 10.85s | n/a |
| 78 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.75s | n/a |
| 79 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 4.8s | n/a |
| 80 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.6s | n/a |
| 81 | ✅ | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 10.28s | n/a |
| 82 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.46s | n/a |
| 83 | ❌ | 0.00 | 0.00 | 1.00 | 0.55 | 0.15 | 13.61s | ✗ behaviour, ✗ retrieval miss |
| 84 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.03s | n/a |
| 85 | ✅ | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 13.63s | n/a |
| 86 | ✅ | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | 6.63s | n/a |
| 87 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 8.78s | n/a |
| 88 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 12.08s | n/a |
| 89 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.12s | n/a |
| 90 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.46s | n/a |
| 91 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 6.8s | n/a |
| 92 | ✅ | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 6.24s | n/a |
| 93 | ✅ | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 7.61s | n/a |
| 94 | ✅ | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 13.37s | n/a |
| 95 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 7.23s | n/a |
| 96 | ✅ | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 7.79s | n/a |
| 97 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.48s | n/a |
| 98 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.95s | n/a |
| 99 | ✅ | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 12.43s | n/a |
| 100 | ✅ | 1.00 | 1.00 | 1.00 | 0.85 | 0.80 | 13.56s | n/a |