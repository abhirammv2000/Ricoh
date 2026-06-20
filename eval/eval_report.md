# 📊 RicohLibrary — Quality Evaluation Report

**Generated:** 2026-06-19 22:11:24  
**LLM judge:** enabled  
**Questions:** 10  

## Aggregate metrics

| Metric | Score | What it means |
|---|---|---|
| Behaviour-match rate | 100% | Answered vs. refused as expected |
| Mean retrieval recall@k | 0.78 | Expected source docs that were retrieved |
| Mean citation precision | 1.00 | Cited docs that are real (no fabricated cites) |
| Mean groundedness | 0.98 | Claims supported by evidence (no hallucination) |
| Mean correctness | 0.97 | Conveys expected facts / refuses correctly |
| Mean latency | 17.37s | Per-question wall-clock |

## Per-question results

| # | Behaviour ✓ | Recall@k | Cite prec. | Grounded | Correct | Latency | Flags |
|---|---|---|---|---|---|---|---|
| 1 | ✅ | 0.50 | 1.00 | 1.00 | 1.00 | 11.03s | — |
| 2 | ✅ | 0.00 | n/a | 1.00 | 1.00 | 15.11s | — |
| 3 | ✅ | n/a | 1.00 | 1.00 | 1.00 | 29.7s | — |
| 4 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 20.69s | — |
| 5 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 12.8s | — |
| 6 | ✅ | 1.00 | 1.00 | 0.97 | 1.00 | 19.67s | — |
| 7 | ✅ | 1.00 | 1.00 | 0.95 | 1.00 | 17.65s | — |
| 8 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 21.92s | — |
| 9 | ✅ | 0.50 | 1.00 | 0.85 | 0.70 | 13.82s | — |
| 10 | ✅ | 1.00 | 1.00 | 1.00 | 1.00 | 11.28s | — |