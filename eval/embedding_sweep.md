# Embedding sweep (retrieval only)

Retriever metrics with the embedding model, candidate pool and reranker
varied. No LLM, so these are exact and free to reproduce. `minilm` is the
current production default.

Decide on `dev` (70 questions), confirm on `holdout` (30). `all` is for
comparison with the README, whose retriever numbers are over all 100.

| model | top_k | rerank | split | R@1 | R@3 | R@5 | MRR | nDCG@5 | missed |
|---|---|---|---|---|---|---|---|---|---|
| bge-base | 10 | no | dev | 0.77 | 0.89 | 0.93 | 0.83 | 0.86 |  |
| bge-base | 10 | no | holdout | 0.73 | 0.87 | 0.90 | 0.81 | 0.83 |  |
| bge-base | 10 | no | all | 0.76 | 0.88 | 0.92 | 0.83 | 0.85 | 8 |
| bge-base | 10 | yes | dev | 0.79 | 0.96 | 0.99 | 0.87 | 0.90 |  |
| bge-base | 10 | yes | holdout | 0.67 | 0.93 | 0.93 | 0.79 | 0.83 |  |
| bge-base | 10 | yes | all | 0.75 | 0.95 | 0.97 | 0.85 | 0.88 | 3 |
| bge-base | 20 | no | dev | 0.77 | 0.90 | 0.91 | 0.83 | 0.85 |  |
| bge-base | 20 | no | holdout | 0.73 | 0.90 | 0.90 | 0.82 | 0.84 |  |
| bge-base | 20 | no | all | 0.76 | 0.90 | 0.91 | 0.83 | 0.85 | 9 |
| bge-base | 20 | yes | dev | 0.79 | 0.96 | 0.99 | 0.87 | 0.90 |  |
| bge-base | 20 | yes | holdout | 0.67 | 0.93 | 0.93 | 0.79 | 0.83 |  |
| bge-base | 20 | yes | all | 0.75 | 0.95 | 0.97 | 0.85 | 0.88 | 3 |
| bge-small | 10 | no | dev | 0.80 | 0.90 | 0.93 | 0.85 | 0.87 |  |
| bge-small | 10 | no | holdout | 0.77 | 0.93 | 0.93 | 0.85 | 0.87 |  |
| bge-small | 10 | no | all | 0.79 | 0.91 | 0.93 | 0.85 | 0.87 | 7 |
| bge-small | 10 | yes | dev | 0.79 | 0.96 | 0.99 | 0.87 | 0.90 |  |
| bge-small | 10 | yes | holdout | 0.67 | 0.93 | 0.93 | 0.79 | 0.83 |  |
| bge-small | 10 | yes | all | 0.75 | 0.95 | 0.97 | 0.85 | 0.88 | 3 |
| bge-small | 20 | no | dev | 0.80 | 0.91 | 0.93 | 0.86 | 0.87 |  |
| bge-small | 20 | no | holdout | 0.77 | 0.93 | 0.93 | 0.85 | 0.87 |  |
| bge-small | 20 | no | all | 0.79 | 0.92 | 0.93 | 0.85 | 0.87 | 7 |
| bge-small | 20 | yes | dev | 0.79 | 0.96 | 0.99 | 0.87 | 0.90 |  |
| bge-small | 20 | yes | holdout | 0.67 | 0.93 | 0.93 | 0.79 | 0.83 |  |
| bge-small | 20 | yes | all | 0.75 | 0.95 | 0.97 | 0.85 | 0.88 | 3 |
| e5-base-v2 | 10 | no | dev | 0.73 | 0.90 | 0.93 | 0.81 | 0.84 |  |
| e5-base-v2 | 10 | no | holdout | 0.83 | 0.93 | 0.93 | 0.88 | 0.90 |  |
| e5-base-v2 | 10 | no | all | 0.76 | 0.91 | 0.93 | 0.83 | 0.86 | 7 |
| e5-base-v2 | 10 | yes | dev | 0.80 | 0.94 | 0.99 | 0.88 | 0.90 |  |
| e5-base-v2 | 10 | yes | holdout | 0.67 | 0.93 | 0.93 | 0.79 | 0.83 |  |
| e5-base-v2 | 10 | yes | all | 0.76 | 0.94 | 0.97 | 0.85 | 0.88 | 3 |
| e5-base-v2 | 20 | no | dev | 0.71 | 0.91 | 0.91 | 0.81 | 0.83 |  |
| e5-base-v2 | 20 | no | holdout | 0.83 | 0.93 | 0.93 | 0.88 | 0.90 |  |
| e5-base-v2 | 20 | no | all | 0.75 | 0.92 | 0.92 | 0.83 | 0.85 | 8 |
| e5-base-v2 | 20 | yes | dev | 0.79 | 0.96 | 0.99 | 0.87 | 0.90 |  |
| e5-base-v2 | 20 | yes | holdout | 0.67 | 0.93 | 0.93 | 0.79 | 0.83 |  |
| e5-base-v2 | 20 | yes | all | 0.75 | 0.95 | 0.97 | 0.85 | 0.88 | 3 |
| minilm | 10 | no | dev | 0.74 | 0.87 | 0.94 | 0.82 | 0.85 |  |
| minilm | 10 | no | holdout | 0.87 | 0.93 | 0.93 | 0.90 | 0.91 |  |
| minilm | 10 | no | all | 0.78 | 0.89 | 0.94 | 0.85 | 0.87 | 6 |
| minilm | 10 | yes | dev | 0.81 | 0.96 | 0.99 | 0.89 | 0.91 |  |
| minilm | 10 | yes | holdout | 0.67 | 0.93 | 0.93 | 0.79 | 0.83 |  |
| minilm | 10 | yes | all | 0.77 | 0.95 | 0.97 | 0.86 | 0.89 | 3 |
| minilm | 20 | no | dev | 0.74 | 0.90 | 0.91 | 0.82 | 0.84 |  |
| minilm | 20 | no | holdout | 0.87 | 0.90 | 0.93 | 0.89 | 0.90 |  |
| minilm | 20 | no | all | 0.78 | 0.90 | 0.92 | 0.84 | 0.86 | 8 |
| minilm | 20 | yes | dev | 0.81 | 0.96 | 0.99 | 0.89 | 0.91 |  |
| minilm | 20 | yes | holdout | 0.67 | 0.93 | 0.93 | 0.79 | 0.83 |  |
| minilm | 20 | yes | all | 0.77 | 0.95 | 0.97 | 0.86 | 0.89 | 3 |
