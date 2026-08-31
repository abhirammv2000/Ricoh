# Embedding sweep (retrieval only)

Retriever metrics with the embedding model, candidate pool and reranker
varied. No LLM, so these are exact and free to reproduce. `minilm` is the
current production default.

Decide on `dev` (70 questions), confirm on `holdout` (30). `all` is for
comparison with the README, whose retriever numbers are over all 100.

| model | top_k | rerank | split | R@1 | R@3 | R@5 | MRR | nDCG@5 | missed |
|---|---|---|---|---|---|---|---|---|---|
| minilm | 10 | no | dev | 0.74 | 0.87 | 0.94 | 0.82 | 0.85 |  |
| minilm | 10 | no | holdout | 0.87 | 0.93 | 0.93 | 0.90 | 0.91 |  |
| minilm | 10 | no | all | 0.78 | 0.89 | 0.94 | 0.85 | 0.87 | 6 |
| minilm | 20 | no | dev | 0.74 | 0.90 | 0.91 | 0.82 | 0.84 |  |
| minilm | 20 | no | holdout | 0.87 | 0.90 | 0.93 | 0.89 | 0.90 |  |
| minilm | 20 | no | all | 0.78 | 0.90 | 0.92 | 0.84 | 0.86 | 8 |
| bge-small | 10 | no | dev | 0.80 | 0.90 | 0.93 | 0.85 | 0.87 |  |
| bge-small | 10 | no | holdout | 0.77 | 0.93 | 0.93 | 0.85 | 0.87 |  |
| bge-small | 10 | no | all | 0.79 | 0.91 | 0.93 | 0.85 | 0.87 | 7 |
| bge-small | 20 | no | dev | 0.80 | 0.91 | 0.93 | 0.86 | 0.87 |  |
| bge-small | 20 | no | holdout | 0.77 | 0.93 | 0.93 | 0.85 | 0.87 |  |
| bge-small | 20 | no | all | 0.79 | 0.92 | 0.93 | 0.85 | 0.87 | 7 |
