# Citera: Quality Evaluation Report

**Generated:** 2026-08-30 20:39:19  
**Agent model:** `claude-sonnet-4-6`  
**Judge model:** `claude-opus-5`  
**Questions:** 70  

> Every mean carries a 95% percentile-bootstrap confidence interval.
> At this sample size the intervals are wide by construction, they
> are reported so the numbers are not read as more precise than the
> eval set can support.

## Aggregate metrics

| Metric | Mean | 95% CI | What it means |
|---|---|---|---|
| Behaviour-match rate | 1.00 | n/a | Answered vs. refused as expected |
| Evidence recall | 0.96 | [0.90, 1.00] | Expected doc reached the synthesizer, across ALL passes/retries (n=70) |
| Citation precision | 1.00 | [1.00, 1.00] | Cited docs exist in the evidence (catches fabricated filenames only) (n=70) |
| Groundedness | 0.97 | [0.95, 0.98] | Claims supported by evidence (n=70) |
| Correctness | 0.98 | [0.97, 0.99] | Conveys expected facts / refuses correctly (n=70) |
| Mean latency | 10.71s | n/a | Per-question wall-clock (max 31.02s) |

### Retriever in isolation

Single retrieval on the raw question, no planner, no sub-queries,
no entity boost, no retry. Ranks counted over distinct documents.
Comparing this against *Evidence recall* separates a retrieval
failure from a planning failure.

| Depth | Recall |
|---|---|
| recall@1 | 0.74 |
| recall@3 | 0.87 |
| recall@5 | 0.94 |

### Cost and where the time goes

Prices are a dated snapshot (2026-08-01); token counts are
the ground truth and cost is derived from them. Agent cost is what serving
a query costs; judge cost is eval overhead and is never folded into it.

- **Mean cost per query: $0.01642** (max $0.07821)
- Mean 1.04 LLM calls, 2,997 in / 495 out tokens
- Whole-benchmark agent cost $1.1493; judge overhead $1.9212

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 70 | 726.46s | 97.3% | $1.08959 | 94.8% |
| tool_agent | 3 | 15.48s | 2.1% | $0.05975 | 5.2% |
| retrieval | 73 | 4.91s | 0.7% | $0.00000 | 0.0% |
| router | 70 | 0.0s | 0.0% | $0.00000 | 0.0% |

### Worst case (what the means hide)

- **evidence_recall** = 0.00, Q44: Which AFP-related feature do I need to install first before I can use any of the other AFP features in RICOH ProcessDirector?
- **groundedness** = 0.80, Q83: If a job misses its first deadline but then finishes within the second deadline, does the red dot stay in the Deadlines portlet or get cleared?
- **correctness** = 0.75, Q100: If we add a color printing line using an InfoPrint 5000, can we still run those color jobs on our existing InfoPrint 4100 or InfoPrint 4000 printers as a fallback?

## Per-question results

| # | Behaviour | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 31 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.2s | - |
| 32 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.6s | - |
| 33 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 8.37s | - |
| 34 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 11.23s | - |
| 35 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 6.7s | - |
| 36 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.88s | - |
| 37 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.26s | - |
| 38 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.95s | - |
| 39 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.11s | - |
| 40 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 5.05s | - |
| 41 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.26s | - |
| 42 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 9.1s | - |
| 43 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 11.62s | - |
| 44 |  | 0.00 | 0.00 | 1.00 | 0.85 | 0.80 | 5.75s | retrieval miss |
| 45 |  | 1.00 | 1.00 | 1.00 | 0.94 | 1.00 | 10.09s | - |
| 46 |  | 1.00 | 1.00 | 1.00 | 0.82 | 1.00 | 8.05s | - |
| 47 |  | 1.00 | 1.00 | 1.00 | 0.90 | 1.00 | 16.26s | - |
| 48 |  | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 15.24s | - |
| 49 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 14.76s | - |
| 50 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.54s | - |
| 51 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.81s | - |
| 52 |  | 1.00 | 1.00 | 1.00 | 0.95 | 0.95 | 11.67s | - |
| 53 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 10.62s | - |
| 54 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 13.35s | - |
| 55 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 8.84s | - |
| 56 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 4.78s | - |
| 57 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 14.51s | - |
| 58 |  | 1.00 | 0.00 | 1.00 | 0.95 | 1.00 | 27.01s | - |
| 59 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.63s | - |
| 60 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 12.44s | - |
| 61 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.05s | - |
| 62 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.67s | - |
| 63 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.1s | - |
| 64 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 7.28s | - |
| 65 |  | 1.00 | 1.00 | 1.00 | 0.95 | 0.85 | 9.87s | - |
| 66 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.82s | - |
| 67 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.87s | - |
| 68 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 16.88s | - |
| 69 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 19.42s | - |
| 70 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 8.66s | - |
| 71 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.58s | - |
| 72 |  | 0.00 | 0.00 | 1.00 | 0.97 | 0.85 | 10.09s | retrieval miss |
| 73 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 10.98s | - |
| 74 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.37s | - |
| 75 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.45s | - |
| 76 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 10.59s | - |
| 77 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 11.04s | - |
| 78 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.37s | - |
| 79 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 5.02s | - |
| 80 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.31s | - |
| 81 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.87s | - |
| 82 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.9s | - |
| 83 |  | 0.00 | 0.00 | 1.00 | 0.80 | 0.90 | 8.61s | retrieval miss |
| 84 |  | 1.00 | 1.00 | 1.00 | 0.97 | 0.95 | 12.88s | - |
| 85 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 13.68s | - |
| 86 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.0s | - |
| 87 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 8.76s | - |
| 88 |  | 1.00 | 1.00 | 1.00 | 0.92 | 0.97 | 15.81s | - |
| 89 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 31.02s | - |
| 90 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.24s | - |
| 91 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.25s | - |
| 92 |  | 1.00 | 1.00 | 1.00 | 0.90 | 1.00 | 6.36s | - |
| 93 |  | 1.00 | 1.00 | 1.00 | 0.85 | 1.00 | 7.36s | - |
| 94 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 11.75s | - |
| 95 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.39s | - |
| 96 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 8.69s | - |
| 97 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.27s | - |
| 98 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.8s | - |
| 99 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 12.76s | - |
| 100 |  | 1.00 | 1.00 | 1.00 | 0.90 | 0.75 | 15.08s | - |