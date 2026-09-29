# Azure AI Search vs local ChromaDB + BM25 (retrieval only)

Same 100 questions, same 1,322 chunks, same MiniLM embeddings, same scoring as
`eval/sweep_embeddings.py`. No LLM, so it is free to reproduce. Decide on `dev`
(70 questions), confirm on `holdout` (30). `all` is over the 100.

Azure semantic ranker requests sent in this run: 200.

| retriever | split | R@1 | R@3 | R@5 | MRR | nDCG@5 | missed | ms/query |
|---|---|---|---|---|---|---|---|---|
| local hybrid (RRF) top_k=10 | dev | 0.74 | 0.87 | 0.94 | 0.82 | 0.85 |  |  |
| local hybrid (RRF) top_k=10 | holdout | 0.87 | 0.93 | 0.93 | 0.90 | 0.91 |  |  |
| local hybrid (RRF) top_k=10 | all | 0.78 | 0.89 | 0.94 | 0.85 | 0.87 | 6 | 283.0 |
| local hybrid (RRF) top_k=20 | dev | 0.74 | 0.90 | 0.91 | 0.82 | 0.84 |  |  |
| local hybrid (RRF) top_k=20 | holdout | 0.87 | 0.90 | 0.93 | 0.89 | 0.90 |  |  |
| local hybrid (RRF) top_k=20 | all | 0.78 | 0.90 | 0.92 | 0.84 | 0.86 | 8 | 477.0 |
| local hybrid (RRF) + cross-encoder top_k=20 | dev | 0.81 | 0.96 | 0.99 | 0.89 | 0.91 |  |  |
| local hybrid (RRF) + cross-encoder top_k=20 | holdout | 0.67 | 0.93 | 0.93 | 0.79 | 0.83 |  |  |
| local hybrid (RRF) + cross-encoder top_k=20 | all | 0.77 | 0.95 | 0.97 | 0.86 | 0.89 | 3 | 6243.0 |
| azure keyword | dev | 0.71 | 0.90 | 0.94 | 0.80 | 0.84 |  |  |
| azure keyword | holdout | 0.77 | 0.93 | 0.93 | 0.84 | 0.87 |  |  |
| azure keyword | all | 0.73 | 0.91 | 0.94 | 0.82 | 0.85 | 6 | 86.8 |
| azure vector | dev | 0.54 | 0.79 | 0.87 | 0.67 | 0.72 |  |  |
| azure vector | holdout | 0.80 | 0.90 | 0.90 | 0.84 | 0.86 |  |  |
| azure vector | all | 0.62 | 0.82 | 0.88 | 0.72 | 0.76 | 12 | 459.8 |
| azure hybrid top_k=10 | dev | 0.73 | 0.89 | 0.90 | 0.81 | 0.83 |  |  |
| azure hybrid top_k=10 | holdout | 0.83 | 0.90 | 0.90 | 0.87 | 0.88 |  |  |
| azure hybrid top_k=10 | all | 0.76 | 0.89 | 0.90 | 0.82 | 0.84 | 10 | 530.4 |
| azure hybrid top_k=20 | dev | 0.73 | 0.90 | 0.91 | 0.81 | 0.84 |  |  |
| azure hybrid top_k=20 | holdout | 0.83 | 0.90 | 0.90 | 0.87 | 0.88 |  |  |
| azure hybrid top_k=20 | all | 0.76 | 0.90 | 0.91 | 0.83 | 0.85 | 9 | 529.4 |
| azure hybrid + semantic ranker top_k=10 | dev | 0.87 | 0.99 | 0.99 | 0.92 | 0.94 |  |  |
| azure hybrid + semantic ranker top_k=10 | holdout | 0.80 | 0.93 | 0.93 | 0.87 | 0.88 |  |  |
| azure hybrid + semantic ranker top_k=10 | all | 0.85 | 0.97 | 0.97 | 0.91 | 0.92 | 3 | 800.7 |
| azure hybrid + semantic ranker top_k=20 | dev | 0.87 | 0.99 | 0.99 | 0.92 | 0.94 |  |  |
| azure hybrid + semantic ranker top_k=20 | holdout | 0.80 | 0.93 | 0.93 | 0.87 | 0.88 |  |  |
| azure hybrid + semantic ranker top_k=20 | all | 0.85 | 0.97 | 0.97 | 0.91 | 0.92 | 3 | 803.6 |
