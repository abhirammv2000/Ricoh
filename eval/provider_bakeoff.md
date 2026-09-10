# Cross-provider bakeoff

**Generated:** 2026-09-10 16:15:35  
**Judge:** `claude-opus-5` (constant across providers)  
**Questions:** 20, sampled from the generated dev split  

Config A (retrieve -> synthesize). Retrieval is shared and identical, so differences are the synthesizer model.

| provider | model | cost/query | latency | out tok | grounded | correct | evid. recall | behaviour |
|---|---|---|---|---|---|---|---|---|
| anthropic | `claude-sonnet-4-6` | $0.01510 | 9.944s | 476 | 0.98 | 0.9925 | 1.0 | 1.0 |
| openai | `gpt-4o-mini` | $0.00050 | 2.609s | 146 | 0.962 | 0.779 | 1.0 | 0.9 |
| google | `gemini-3.6-flash` | $0.00030 | 5.2065s | 240 | 0.9975 | 0.9525 | 1.0 | 1.0 |

Baseline is `anthropic`. A cheaper model that holds groundedness and correctness within the judge's ~0.10 noise floor would be a real cost lever; one that drops either is not.
