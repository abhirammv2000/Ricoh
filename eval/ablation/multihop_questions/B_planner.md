# Citera: Quality Evaluation Report

**Generated:** 2026-09-10 15:41:43  
**Agent model:** `claude-sonnet-4-6`  
**Judge model:** `claude-opus-5`  
**Questions:** 20  

> Every mean carries a 95% percentile-bootstrap confidence interval.
> At this sample size the intervals are wide by construction, they
> are reported so the numbers are not read as more precise than the
> eval set can support.

## Aggregate metrics

| Metric | Mean | 95% CI | What it means |
|---|---|---|---|
| Behaviour-match rate | 1.00 | n/a | Answered vs. refused as expected |
| Evidence recall | 0.82 | [0.70, 0.95] | Expected doc reached the synthesizer, across ALL passes/retries (n=20) |
| Citation precision | 1.00 | [1.00, 1.00] | Cited docs exist in the evidence (catches fabricated filenames only) (n=20) |
| Groundedness | 0.98 | [0.97, 0.99] | Claims supported by evidence (n=20) |
| Correctness | 0.92 | [0.85, 0.97] | Conveys expected facts / refuses correctly (n=20) |
| Mean latency | 25.43s | n/a | Per-question wall-clock (max 45.16s) |

### Retriever in isolation

Single retrieval on the raw question, no planner, no sub-queries,
no entity boost, no retry. Ranks counted over distinct documents.
Comparing this against *Evidence recall* separates a retrieval
failure from a planning failure.

| Depth | Recall |
|---|---|
| recall@1 | 0.28 |
| recall@3 | 0.62 |
| recall@5 | 0.78 |

### Cost and where the time goes

Prices are a dated snapshot (2026-08-01); token counts are
the ground truth and cost is derived from them. Agent cost is what serving
a query costs; judge cost is eval overhead and is never folded into it.

- **Mean cost per query: $0.03172** (max $0.05532)
- Mean 2.0 LLM calls, 5,802 in / 954 out tokens
- Whole-benchmark agent cost $0.6344; judge overhead $1.0197

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 20 | 388.0s | 76.3% | $0.59525 | 93.8% |
| planner | 20 | 100.92s | 19.9% | $0.03916 | 6.2% |
| retrieval | 90 | 19.3s | 3.8% | $0.00000 | 0.0% |
| citation_guardrail | 20 | 0.0s | 0.0% | $0.00000 | 0.0% |

### Worst case (what the means hide)

- **evidence_recall** = 0.00, Q8: How do I create a custom document property in RICOH ProcessDirector, and what do I have to do with it before I can use it to sort or split AFP documents?
- **groundedness** = 0.95, Q8: How do I create a custom document property in RICOH ProcessDirector, and what do I have to do with it before I can use it to sort or split AFP documents?
- **correctness** = 0.45, Q12: The PDF Document Support feature uses RICOH ProcessDirector Plug-in for Adobe Acrobat. What does it let you do with the individual documents in a PDF file, and where is the markup saved?

## Per-question results

| # | Behaviour | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 23.56s | - |
| 2 |  | 0.50 | 0.50 | 1.00 | 1.00 | 0.95 | 20.08s | - |
| 3 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 17.83s | - |
| 4 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 26.69s | - |
| 5 |  | 1.00 | 0.50 | 1.00 | 0.97 | 1.00 | 26.66s | - |
| 6 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 16.2s | - |
| 7 |  | 1.00 | 1.00 | 1.00 | 0.97 | 0.93 | 21.06s | - |
| 8 |  | 0.00 | 0.00 | 1.00 | 0.95 | 0.85 | 28.45s | retrieval miss |
| 9 |  | 0.50 | 0.50 | 1.00 | 0.95 | 0.95 | 21.49s | - |
| 10 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 22.7s | - |
| 11 |  | 1.00 | 1.00 | 1.00 | 0.95 | 0.90 | 27.91s | - |
| 12 |  | 0.50 | 0.50 | 1.00 | 0.95 | 0.45 | 25.05s | - |
| 13 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 36.93s | - |
| 14 |  | 1.00 | 0.50 | 1.00 | 0.98 | 0.97 | 31.43s | - |
| 15 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 24.09s | - |
| 16 |  | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 22.05s | - |
| 17 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 17.43s | - |
| 18 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 45.16s | - |
| 19 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 23.98s | - |
| 20 |  | 0.50 | 0.50 | 1.00 | 1.00 | 0.65 | 29.88s | - |