# Citera: Quality Evaluation Report

**Generated:** 2026-08-30 20:21:40  
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
| Citation precision | 1.00 | [0.99, 1.00] | Cited docs exist in the evidence (catches fabricated filenames only) (n=70) |
| Groundedness | 0.97 | [0.96, 0.98] | Claims supported by evidence (n=70) |
| Correctness | 0.99 | [0.98, 1.00] | Conveys expected facts / refuses correctly (n=70) |
| Mean latency | 16.49s | n/a | Per-question wall-clock (max 27.43s) |

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

- **Mean cost per query: $0.04140** (max $0.07706)
- Mean 3.0 LLM calls, 10,691 in / 622 out tokens
- Whole-benchmark agent cost $2.8979; judge overhead $3.1460

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 70 | 788.66s | 68.4% | $1.65957 | 57.3% |
| planner | 70 | 212.77s | 18.4% | $0.14917 | 5.1% |
| verifier | 70 | 128.84s | 11.2% | $1.08914 | 37.6% |
| retrieval | 393 | 23.0s | 2.0% | $0.00000 | 0.0% |

### Worst case (what the means hide)

- **evidence_recall** = 1.00, Q31: When converting line data to AFP, what information does a page definition control during composition?
- **groundedness** = 0.85, Q47: What's the difference between a Kodak PDF printer device and a Passthrough printer device, and which one should I use if my printer model is supported by both?
- **correctness** = 0.70, Q75: When setting up a Xerox PDF printer for banner pages, what happens if the Header copies and Trailer copies job properties are set to 0, even if banner pages are turned on at the printer level?

## Per-question results

| # | Behaviour | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 31 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.23s | - |
| 32 |  | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 12.38s | - |
| 33 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.92s | - |
| 34 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 20.93s | - |
| 35 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 13.31s | - |
| 36 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 15.05s | - |
| 37 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 15.57s | - |
| 38 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.77s | - |
| 39 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.73s | - |
| 40 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.79s | - |
| 41 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 13.25s | - |
| 42 |  | 1.00 | 1.00 | 1.00 | 0.90 | 1.00 | 15.46s | - |
| 43 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 15.57s | - |
| 44 |  | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 15.04s | - |
| 45 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 17.03s | - |
| 46 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 20.26s | - |
| 47 |  | 1.00 | 1.00 | 1.00 | 0.85 | 1.00 | 22.32s | - |
| 48 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 24.58s | - |
| 49 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 24.0s | - |
| 50 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 12.29s | - |
| 51 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 13.97s | - |
| 52 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 16.76s | - |
| 53 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 16.13s | - |
| 54 |  | 1.00 | 1.00 | 0.75 | 0.93 | 1.00 | 19.68s | - |
| 55 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 15.33s | - |
| 56 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.91s | - |
| 57 |  | 1.00 | 1.00 | 1.00 | 0.94 | 1.00 | 19.79s | - |
| 58 |  | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 15.32s | - |
| 59 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 15.13s | - |
| 60 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 22.13s | - |
| 61 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.9s | - |
| 62 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 17.45s | - |
| 63 |  | 1.00 | 1.00 | 1.00 | 0.97 | 0.93 | 16.7s | - |
| 64 |  | 1.00 | 1.00 | 1.00 | 0.95 | 0.85 | 13.17s | - |
| 65 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 14.81s | - |
| 66 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 17.03s | - |
| 67 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 15.09s | - |
| 68 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 22.41s | - |
| 69 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 21.84s | - |
| 70 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 14.89s | - |
| 71 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 16.3s | - |
| 72 |  | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 27.43s | - |
| 73 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.61s | - |
| 74 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 17.18s | - |
| 75 |  | 1.00 | 1.00 | 1.00 | 0.92 | 0.70 | 13.54s | - |
| 76 |  | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 19.56s | - |
| 77 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 18.88s | - |
| 78 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 20.29s | - |
| 79 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.77s | - |
| 80 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.35s | - |
| 81 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 17.49s | - |
| 82 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 20.24s | - |
| 83 |  | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 12.44s | - |
| 84 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 20.8s | - |
| 85 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 18.35s | - |
| 86 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 13.48s | - |
| 87 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 13.6s | - |
| 88 |  | 1.00 | 1.00 | 1.00 | 0.90 | 0.97 | 23.91s | - |
| 89 |  | 1.00 | 1.00 | 1.00 | 0.97 | 0.92 | 14.57s | - |
| 90 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 18.78s | - |
| 91 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 15.04s | - |
| 92 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 13.31s | - |
| 93 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 12.49s | - |
| 94 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 18.41s | - |
| 95 |  | 1.00 | 1.00 | 1.00 | 0.97 | 0.95 | 12.97s | - |
| 96 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 11.32s | - |
| 97 |  | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 15.16s | - |
| 98 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.63s | - |
| 99 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 16.53s | - |
| 100 |  | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 20.66s | - |