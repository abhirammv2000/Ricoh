# Query rewriting before retrieval: HyDE and multi-query

Retrieval only, no judge. MiniLM index, hybrid retrieval (vector + BM25, RRF), top_k 10, final_k 5,
no reranker, 100 questions (70 dev, 30 holdout). The LLM for the rewrites is Gemini flash;
its outputs are cached in `eval/query_transform_cache.json`, so this reproduces exactly.
Decide on dev and confirm on holdout, as in the embedding sweep.

| config | split | R@1 | R@3 | R@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| baseline (raw question) | dev | 0.743 | 0.871 | 0.943 | 0.825 | 0.855 |
| baseline (raw question) | holdout | 0.867 | 0.933 | 0.933 | 0.900 | 0.909 |
| baseline (raw question) | all | 0.780 | 0.890 | 0.940 | 0.848 | 0.871 |
| HyDE, passage replaces question for vector search | dev | 0.643 | 0.871 | 0.957 | 0.759 | 0.808 |
| HyDE, passage replaces question for vector search | holdout | 0.667 | 0.900 | 0.933 | 0.779 | 0.818 |
| HyDE, passage replaces question for vector search | all | 0.650 | 0.880 | 0.950 | 0.765 | 0.811 |
| HyDE added as a third list | dev | 0.700 | 0.914 | 0.929 | 0.800 | 0.833 |
| HyDE added as a third list | holdout | 0.800 | 0.867 | 0.900 | 0.842 | 0.856 |
| HyDE added as a third list | all | 0.730 | 0.900 | 0.920 | 0.813 | 0.840 |
| multi-query (3 rewrites + original) | dev | 0.700 | 0.900 | 0.929 | 0.795 | 0.829 |
| multi-query (3 rewrites + original) | holdout | 0.833 | 0.933 | 0.933 | 0.883 | 0.896 |
| multi-query (3 rewrites + original) | all | 0.740 | 0.910 | 0.930 | 0.822 | 0.849 |

Gemini's safety filter blocked a few harmless requests (no output returned). For those questions the
raw question is used instead, so they behave like the baseline. HyDE blocked on ids ['17', '22'];
multi-query blocked on ids ['16', '22', '41', '45'].

Questions whose recall@5 changed against the baseline (out of 100):

| config | better | worse |
|---|---|---|
| HyDE, passage replaces question for vector search | 2 | 1 |
| HyDE added as a third list | 1 | 3 |
| multi-query (3 rewrites + original) | 2 | 3 |
