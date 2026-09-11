# Reranker sweep (retrieval only)

Reranker model varied against the `minilm` index, `top_k=20 -> final_k=5`, over the 100-question set. No LLM. `none` is the fused pool with no reranking.

| reranker | split | R@1 | R@3 | R@5 | MRR | nDCG@5 | missed | s/query |
|---|---|---|---|---|---|---|---|---|
| `none` | dev | 0.74 | 0.90 | 0.91 | 0.82 | 0.84 |  |  |
| `none` | holdout | 0.87 | 0.90 | 0.93 | 0.89 | 0.90 |  |  |
| `none` | all | 0.78 | 0.90 | 0.92 | 0.84 | 0.86 | 8 |  |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | dev | 0.81 | 0.96 | 0.99 | 0.89 | 0.91 |  | 5.32 |
|  | holdout | 0.67 | 0.93 | 0.93 | 0.79 | 0.83 |  |  |
|  | all | 0.77 | 0.95 | 0.97 | 0.86 | 0.89 | 3 |  |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | dev | 0.79 | 0.97 | 0.99 | 0.88 | 0.90 |  | 9.41 |
|  | holdout | 0.77 | 0.93 | 0.93 | 0.84 | 0.86 |  |  |
|  | all | 0.78 | 0.96 | 0.97 | 0.87 | 0.89 | 3 |  |
| `BAAI/bge-reranker-base` | dev | 0.83 | 0.97 | 0.99 | 0.89 | 0.91 |  | 26.87 |
|  | holdout | 0.77 | 0.93 | 0.93 | 0.84 | 0.87 |  |  |
|  | all | 0.81 | 0.96 | 0.97 | 0.88 | 0.90 | 3 |  |
| `BAAI/bge-reranker-v2-m3` | dev | 0.90 | 0.99 | 0.99 | 0.94 | 0.95 |  | 120.54 |
|  | holdout | 0.77 | 0.93 | 0.93 | 0.85 | 0.87 |  |  |
|  | all | 0.86 | 0.97 | 0.97 | 0.91 | 0.93 | 3 |  |
