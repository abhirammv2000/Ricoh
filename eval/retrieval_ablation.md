# Retrieval ablation: each method alone, and the fusion method

Retrieval only, no LLM, top_k=10, final_k=5, the 100 questions (2 of them contain a code-like token).

| method | R@1 | R@3 | R@5 | MRR | missed | dev R@5 | holdout R@5 |
|---|---|---|---|---|---|---|---|
| vector only | 0.62 | 0.82 | 0.88 | 0.72 | 12 | 0.871 | 0.900 |
| BM25 only | 0.68 | 0.86 | 0.88 | 0.77 | 12 | 0.857 | 0.933 |
| RRF k=60 (current) | 0.78 | 0.89 | 0.94 | 0.84 | 6 | 0.943 | 0.933 |
| RRF k=10 | 0.79 | 0.89 | 0.94 | 0.85 | 6 | 0.943 | 0.933 |
| RRF k=30 | 0.78 | 0.89 | 0.94 | 0.84 | 6 | 0.943 | 0.933 |
| RRF k=100 | 0.78 | 0.89 | 0.94 | 0.84 | 6 | 0.943 | 0.933 |
| score blend 50/50 | 0.75 | 0.92 | 0.94 | 0.83 | 6 | 0.943 | 0.933 |
| score blend 70/30 vector | 0.71 | 0.90 | 0.94 | 0.81 | 6 | 0.943 | 0.933 |
| score blend 30/70 vector | 0.76 | 0.89 | 0.92 | 0.83 | 8 | 0.914 | 0.933 |

BM25 alone misses 9 questions that vector search finds, and vector search misses 9 that BM25 finds. Both miss 3.
RRF misses 6: those 3 and [58, 72, 83], which one of the two methods alone finds.

## Exact codes from the corpus

The corpus has 18 distinct code-like tokens in 1322 chunks. 14 of them are in at most 2 chunks, and each was looked up as "What does <code> mean?".

| method | found in the top 5 |
|---|---|
| vector only | 6 of 14 |
| BM25 only | 14 of 14 |
| RRF (current) | 12 of 14 |
