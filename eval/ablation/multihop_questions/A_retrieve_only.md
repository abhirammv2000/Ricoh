# Citera: Quality Evaluation Report

**Generated:** 2026-09-10 15:31:10  
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
| Evidence recall | 0.78 | [0.65, 0.90] | Expected doc reached the synthesizer, across ALL passes/retries (n=20) |
| Citation precision | 1.00 | [1.00, 1.00] | Cited docs exist in the evidence (catches fabricated filenames only) (n=20) |
| Groundedness | 0.97 | [0.96, 0.99] | Claims supported by evidence (n=20) |
| Correctness | 0.91 | [0.85, 0.96] | Conveys expected facts / refuses correctly (n=20) |
| Mean latency | 16.0s | n/a | Per-question wall-clock (max 29.6s) |

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

- **Mean cost per query: $0.02050** (max $0.03034)
- Mean 1.0 LLM calls, 3,007 in / 765 out tokens
- Whole-benchmark agent cost $0.4100; judge overhead $0.6533

| Stage | Calls | LLM seconds | % of time | Cost | % of cost |
|---|---|---|---|---|---|
| synthesizer | 20 | 315.32s | 99.1% | $0.40999 | 100.0% |
| retrieval | 20 | 2.9s | 0.9% | $0.00000 | 0.0% |
| citation_guardrail | 20 | 0.0s | 0.0% | $0.00000 | 0.0% |

### Worst case (what the means hide)

- **evidence_recall** = 0.00, Q8: How do I create a custom document property in RICOH ProcessDirector, and what do I have to do with it before I can use it to sort or split AFP documents?
- **groundedness** = 0.90, Q20: What does the Reports feature add to RICOH ProcessDirector, and which data collectors can I configure with it?
- **correctness** = 0.55, Q12: The PDF Document Support feature uses RICOH ProcessDirector Plug-in for Adobe Acrobat. What does it let you do with the individual documents in a PDF file, and where is the markup saved?

## Per-question results

| # | Behaviour | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.72s | - |
| 2 |  | 0.50 | 0.50 | 1.00 | 1.00 | 0.95 | 10.34s | - |
| 3 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 11.75s | - |
| 4 |  | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 18.86s | - |
| 5 |  | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 17.7s | - |
| 6 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 13.38s | - |
| 7 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.85 | 15.96s | - |
| 8 |  | 0.00 | 0.00 | 1.00 | 0.92 | 0.85 | 23.13s | retrieval miss |
| 9 |  | 0.50 | 0.50 | 1.00 | 0.93 | 0.88 | 12.25s | - |
| 10 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12.36s | - |
| 11 |  | 1.00 | 1.00 | 1.00 | 0.92 | 0.85 | 15.95s | - |
| 12 |  | 0.50 | 0.50 | 1.00 | 1.00 | 0.55 | 13.94s | - |
| 13 |  | 1.00 | 1.00 | 1.00 | 0.95 | 0.85 | 13.37s | - |
| 14 |  | 0.50 | 0.50 | 1.00 | 1.00 | 0.90 | 17.82s | - |
| 15 |  | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 19.01s | - |
| 16 |  | 0.50 | 0.50 | 1.00 | 0.95 | 1.00 | 13.52s | - |
| 17 |  | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 14.28s | - |
| 18 |  | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 29.6s | - |
| 19 |  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 18.66s | - |
| 20 |  | 0.50 | 0.50 | 1.00 | 0.90 | 0.60 | 16.37s | - |