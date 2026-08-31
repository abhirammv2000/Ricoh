# Citera: Quality Evaluation Report

**Generated:** 2026-08-30 19:37:07  
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
| Behaviour-match rate | 0.99 | n/a | Answered vs. refused as expected |
| Evidence recall | 0.94 | [0.89, 0.99] | Expected doc reached the synthesizer, across ALL passes/retries (n=70) |
| Citation precision | 1.00 | [1.00, 1.00] | Cited docs exist in the evidence (catches fabricated filenames only) (n=70) |
| Groundedness | 0.96 | [0.95, 0.97] | Claims supported by evidence (n=70) |
| Correctness | 0.97 | [0.95, 0.99] | Conveys expected facts / refuses correctly (n=70) |
| Mean latency | 10.23s | n/a | Per-question wall-clock (max 18.16s) |

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

- **Mean cost per query: $0.01563** (max $0.02237)
- Mean 1.0 LLM calls, 2,759 in / 490 out tokens
- Whole-benchmark agent cost $1.0941; judge overhead $1.8991

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 70 | 711.07s | 99.5% | $1.09409 | 100.0% |
| retrieval | 70 | 3.53s | 0.5% | $0.00000 | 0.0% |

### Worst case (what the means hide)

- **evidence_recall** = 0.00, Q44: Which AFP-related feature do I need to install first before I can use any of the other AFP features in RICOH ProcessDirector?
- **groundedness** = 0.75, Q47: What's the difference between a Kodak PDF printer device and a Passthrough printer device, and which one should I use if my printer model is supported by both?
- **correctness** = 0.60, Q58: After finishing the page group and document property definitions in the Acrobat plug-in, where does the control file need to be sent, and what permission is required for that location?

## Per-question results

| # | Behaviour | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 31 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.02s | - |
| 32 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 7.28s | - |
| 33 |  | 1.00 | 1.00 | 1.00 | 0.90 | 1.00 | 8.06s | - |
| 34 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 11.04s | - |
| 35 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 6.36s | - |
| 36 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.22s | - |
| 37 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.66s | - |
| 38 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.86s | - |
| 39 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 5.84s | - |
| 40 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 4.93s | - |
| 41 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 9.07s | - |
| 42 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 9.5s | - |
| 43 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 10.13s | - |
| 44 |  | 0.00 | 0.00 | 1.00 | 0.95 | 0.80 | 6.45s | retrieval miss |
| 45 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.04s | - |
| 46 |  | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 | 9.5s | - |
| 47 |  | 1.00 | 1.00 | 1.00 | 0.75 | 0.95 | 16.14s | - |
| 48 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 13.65s | - |
| 49 |  | 1.00 | 1.00 | 1.00 | 0.95 | 0.95 | 14.71s | - |
| 50 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.08s | - |
| 51 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.49s | - |
| 52 |  | 1.00 | 1.00 | 1.00 | 0.92 | 0.95 | 13.96s | - |
| 53 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 12.89s | - |
| 54 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 12.38s | - |
| 55 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 8.62s | - |
| 56 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 4.54s | - |
| 57 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 13.42s | - |
| 58 |  | 0.00 | 0.00 | 1.00 | 0.95 | 0.60 | 10.12s | behaviour, retrieval miss |
| 59 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 | 9.91s | - |
| 60 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 13.04s | - |
| 61 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 9.06s | - |
| 62 |  | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 9.75s | - |
| 63 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 13.37s | - |
| 64 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.15s | - |
| 65 |  | 1.00 | 1.00 | 1.00 | 0.85 | 0.85 | 9.04s | - |
| 66 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.51s | - |
| 67 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.11s | - |
| 68 |  | 1.00 | 1.00 | 1.00 | 0.97 | 0.90 | 14.88s | - |
| 69 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 18.16s | - |
| 70 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 10.72s | - |
| 71 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.98s | - |
| 72 |  | 0.00 | 0.00 | 1.00 | 1.00 | 0.85 | 14.78s | retrieval miss |
| 73 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.98s | - |
| 74 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.63s | - |
| 75 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 6.96s | - |
| 76 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.7s | - |
| 77 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 9.99s | - |
| 78 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 14.88s | - |
| 79 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 7.33s | - |
| 80 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.86s | - |
| 81 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.86s | - |
| 82 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.31s | - |
| 83 |  | 0.00 | 0.00 | 1.00 | 0.80 | 0.95 | 10.69s | retrieval miss |
| 84 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.39s | - |
| 85 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 12.99s | - |
| 86 |  | 1.00 | 1.00 | 1.00 | 0.90 | 1.00 | 8.75s | - |
| 87 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.08s | - |
| 88 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 14.2s | - |
| 89 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 7.88s | - |
| 90 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.09s | - |
| 91 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 7.01s | - |
| 92 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 6.6s | - |
| 93 |  | 1.00 | 1.00 | 1.00 | 0.85 | 1.00 | 8.52s | - |
| 94 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 9.95s | - |
| 95 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 8.65s | - |
| 96 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 7.77s | - |
| 97 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.16s | - |
| 98 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 13.02s | - |
| 99 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 11.83s | - |
| 100 |  | 1.00 | 1.00 | 1.00 | 0.85 | 0.75 | 16.46s | - |