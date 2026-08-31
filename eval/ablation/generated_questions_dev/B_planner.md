# Citera: Quality Evaluation Report

**Generated:** 2026-08-30 19:57:59  
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
| Evidence recall | 1.00 | [1.00, 1.00] | Expected doc reached the synthesizer, across ALL passes/retries (n=70) |
| Citation precision | 0.99 | [0.98, 1.00] | Cited docs exist in the evidence (catches fabricated filenames only) (n=70) |
| Groundedness | 0.98 | [0.97, 0.98] | Claims supported by evidence (n=70) |
| Correctness | 0.99 | [0.98, 1.00] | Conveys expected facts / refuses correctly (n=70) |
| Mean latency | 14.33s | n/a | Per-question wall-clock (max 23.79s) |

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

- **Mean cost per query: $0.02620** (max $0.04589)
- Mean 2.0 LLM calls, 5,660 in / 615 out tokens
- Whole-benchmark agent cost $1.8343; judge overhead $3.1932

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 70 | 796.77s | 79.5% | $1.68422 | 91.8% |
| planner | 70 | 193.53s | 19.3% | $0.15006 | 8.2% |
| retrieval | 396 | 12.55s | 1.3% | $0.00000 | 0.0% |

### Worst case (what the means hide)

- **evidence_recall** = 1.00, Q31: When converting line data to AFP, what information does a page definition control during composition?
- **groundedness** = 0.90, Q46: When using a CopyToFolder step to send jobs to a printer hot folder, what functionality is lost compared to using a Passthrough printer?
- **correctness** = 0.75, Q75: When setting up a Xerox PDF printer for banner pages, what happens if the Header copies and Trailer copies job properties are set to 0, even if banner pages are turned on at the printer level?

## Per-question results

| # | Behaviour | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 31 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.87s | - |
| 32 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 19.07s | - |
| 33 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.19s | - |
| 34 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 18.33s | - |
| 35 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 13.21s | - |
| 36 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.09s | - |
| 37 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.43s | - |
| 38 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 15.38s | - |
| 39 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.94s | - |
| 40 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.66s | - |
| 41 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 12.78s | - |
| 42 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 12.4s | - |
| 43 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.93s | - |
| 44 |  | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 11.66s | - |
| 45 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.75s | - |
| 46 |  | 1.00 | 1.00 | 0.75 | 0.90 | 1.00 | 14.88s | - |
| 47 |  | 1.00 | 1.00 | 1.00 | 0.90 | 1.00 | 16.04s | - |
| 48 |  | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 20.31s | - |
| 49 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 20.16s | - |
| 50 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 10.7s | - |
| 51 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.42s | - |
| 52 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.75s | - |
| 53 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 13.31s | - |
| 54 |  | 1.00 | 1.00 | 0.75 | 0.92 | 1.00 | 19.82s | - |
| 55 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.58s | - |
| 56 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 9.48s | - |
| 57 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 17.65s | - |
| 58 |  | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 12.09s | - |
| 59 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 15.07s | - |
| 60 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 15.26s | - |
| 61 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.2s | - |
| 62 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.56s | - |
| 63 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.0s | - |
| 64 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 11.92s | - |
| 65 |  | 1.00 | 1.00 | 1.00 | 0.95 | 0.95 | 15.55s | - |
| 66 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 16.09s | - |
| 67 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.72s | - |
| 68 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 23.62s | - |
| 69 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 21.02s | - |
| 70 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.84s | - |
| 71 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.13s | - |
| 72 |  | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 19.29s | - |
| 73 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.14s | - |
| 74 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 16.44s | - |
| 75 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 10.66s | - |
| 76 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.37s | - |
| 77 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 17.05s | - |
| 78 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 19.25s | - |
| 79 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.39s | - |
| 80 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 13.42s | - |
| 81 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.24s | - |
| 82 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 16.8s | - |
| 83 |  | 1.00 | 0.00 | 1.00 | 0.95 | 1.00 | 9.89s | - |
| 84 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 19.67s | - |
| 85 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 18.42s | - |
| 86 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.88s | - |
| 87 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 14.46s | - |
| 88 |  | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 23.79s | - |
| 89 |  | 1.00 | 1.00 | 1.00 | 0.92 | 0.85 | 11.2s | - |
| 90 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 13.42s | - |
| 91 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 11.25s | - |
| 92 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.17s | - |
| 93 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 9.49s | - |
| 94 |  | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 15.79s | - |
| 95 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 12.45s | - |
| 96 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 11.17s | - |
| 97 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 13.22s | - |
| 98 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.53s | - |
| 99 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.05s | - |
| 100 |  | 1.00 | 1.00 | 1.00 | 0.90 | 1.00 | 18.43s | - |