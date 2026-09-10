# Citera: Quality Evaluation Report

**Generated:** 2026-09-10 15:52:06  
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
| Correctness | 0.91 | [0.84, 0.97] | Conveys expected facts / refuses correctly (n=20) |
| Mean latency | 26.26s | n/a | Per-question wall-clock (max 51.33s) |

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

- **Mean cost per query: $0.04850** (max $0.07804)
- Mean 3.0 LLM calls, 11,303 in / 973 out tokens
- Whole-benchmark agent cost $0.9700; judge overhead $1.0263

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 20 | 382.07s | 72.8% | $0.60131 | 62.0% |
| planner | 20 | 76.82s | 14.6% | $0.03923 | 4.0% |
| verifier | 20 | 56.73s | 10.8% | $0.32948 | 34.0% |
| retrieval | 90 | 9.44s | 1.8% | $0.00000 | 0.0% |
| citation_guardrail | 20 | 0.0s | 0.0% | $0.00000 | 0.0% |

### Worst case (what the means hide)

- **evidence_recall** = 0.00, Q8: How do I create a custom document property in RICOH ProcessDirector, and what do I have to do with it before I can use it to sort or split AFP documents?
- **groundedness** = 0.92, Q11: In RICOH Visual Workbench, what is the difference between editing the text value of an index tag itself and editing the text for an index tag that is linked to a property?
- **correctness** = 0.40, Q12: The PDF Document Support feature uses RICOH ProcessDirector Plug-in for Adobe Acrobat. What does it let you do with the individual documents in a PDF file, and where is the markup saved?

## Per-question results

| # | Behaviour | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 35.74s | - |
| 2 |  | 0.50 | 0.50 | 1.00 | 1.00 | 0.95 | 32.17s | - |
| 3 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.07s | - |
| 4 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 28.63s | - |
| 5 |  | 1.00 | 0.50 | 1.00 | 0.95 | 1.00 | 24.57s | - |
| 6 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 17.66s | - |
| 7 |  | 1.00 | 1.00 | 1.00 | 0.97 | 0.93 | 21.19s | - |
| 8 |  | 0.00 | 0.00 | 1.00 | 0.95 | 0.88 | 36.51s | retrieval miss |
| 9 |  | 0.50 | 0.50 | 1.00 | 0.95 | 0.95 | 27.77s | - |
| 10 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 18.04s | - |
| 11 |  | 1.00 | 1.00 | 1.00 | 0.92 | 0.95 | 24.71s | - |
| 12 |  | 0.50 | 0.50 | 1.00 | 1.00 | 0.40 | 24.52s | - |
| 13 |  | 1.00 | 1.00 | 1.00 | 0.95 | 0.80 | 26.22s | - |
| 14 |  | 1.00 | 0.50 | 1.00 | 1.00 | 0.85 | 27.64s | - |
| 15 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 27.57s | - |
| 16 |  | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 20.05s | - |
| 17 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 24.2s | - |
| 18 |  | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 51.33s | - |
| 19 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 21.32s | - |
| 20 |  | 0.50 | 0.50 | 1.00 | 0.95 | 0.60 | 21.37s | - |