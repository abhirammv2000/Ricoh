# Cross-provider bakeoff

**Generated:** 2026-09-12 16:00:32  
**Judge:** `claude-opus-5` (constant across providers)  
**Questions:** 20, sampled from the generated dev split  

Config A (retrieve -> synthesize). Retrieval is shared and identical, so differences are the synthesizer model.

| provider | model | cost/query | latency | out tok | grounded | correct | evid. recall | behaviour |
|---|---|---|---|---|---|---|---|---|
| anthropic | `claude-sonnet-4-6` | $0.01510 | 9.944s | 476 | 0.98 | 0.9925 | 1.0 | 1.0 |
| openai | `gpt-4o-mini` | $0.00050 | 2.609s | 146 | 0.962 | 0.779 | 1.0 | 0.9 |
| google | `gemini-3.6-flash` | $0.00030 | 5.2065s | 240 | 0.9975 | 0.9525 | 1.0 | 1.0 |
| self_hosted | `citera-finetuned` | $0.00000 | 25.197s | 405 | 0.9055 | 0.941 | 1.0 | 1.0 |

Baseline is `anthropic`. A cheaper model that holds groundedness and correctness within the judge's ~0.10 noise floor would be a real cost lever; one that drops either is not.

`self_hosted` is the citera-finetune QLoRA distillation of Llama 3.1 8B (see
`citera-finetune/`), served from a single on-demand GCP L4 rather than a
metered API, so its $0/query is not directly comparable to the other rows: the
real cost is the GPU-hour the instance runs, not a per-token price. On the
judged metrics it lands close to the two cheap commercial models (correctness
0.941, inside the judge's noise floor of anthropic's 0.9925) while running
noticeably slower (25.2s vs 2.6-9.9s), unsurprising for an unoptimized single
L4 with no batching versus provider-side serving infrastructure at scale.
