# Multi-turn follow-up evaluation

**Generated:** 2026-09-10 17:30:09  
**Judge:** `claude-opus-5`  
**Turns:** 36 across 12 chains  

`standalone_target` is the hand-written ideal rewrite. Follow-up turns carry the real prior answers as history.

## Condensation

- Mean cosine(rewrite, target) on follow-ups: **0.921**
- Retriever recall on the raw follow-up: 0.625
- Retriever recall after condensation: **0.875**

The recall lift is the point: a follow-up like "what about those?" retrieves nothing on its own; the rewrite makes it a real query.

## Answer quality, first turn vs follow-ups

| | first turns | follow-ups |
|---|---|---|
| n | 12 | 24 |
| evidence recall | 1.0 | 0.875 |
| behaviour match | 1.0 | 1.0 |
| groundedness | 0.983 | 0.959 |
| correctness | 0.944 | 0.929 |

Agent cost $0.64799, judge overhead $1.0175.

## Per-turn

| chain | turn | rewrite~target | recall raw->condensed | grounded | correct |
|---|---|---|---|---|---|
| error-path | 1 | - | 1.0->1.0 | 1.0 | 1.0 |
| error-path | 2 | 0.966 | 0.0->1.0 | 0.95 | 1.0 |
| error-path | 3 | 0.966 | 1.0->1.0 | 1.0 | 1.0 |
| shutdown | 1 | - | 1.0->1.0 | 1.0 | 1.0 |
| shutdown | 2 | 0.98 | 1.0->1.0 | 1.0 | 1.0 |
| shutdown | 3 | 0.956 | 1.0->1.0 | 0.95 | 1.0 |
| fusionpro | 1 | - | 1.0->1.0 | 1.0 | 1.0 |
| fusionpro | 2 | 1.0 | 0.0->1.0 | 1.0 | 1.0 |
| fusionpro | 3 | 0.72 | 0.0->1.0 | 1.0 | 1.0 |
| pclout | 1 | - | 1.0->1.0 | 0.95 | 1.0 |
| pclout | 2 | 0.854 | 1.0->1.0 | 0.95 | 1.0 |
| pclout | 3 | 0.97 | 0.0->1.0 | 0.85 | 1.0 |
| inserters | 1 | - | 1.0->1.0 | 1.0 | 1.0 |
| inserters | 2 | 0.943 | 1.0->1.0 | 1.0 | 1.0 |
| inserters | 3 | 0.678 | 0.0->1.0 | 0.97 | 1.0 |
| servers-os | 1 | - | 1.0->1.0 | 1.0 | 1.0 |
| servers-os | 2 | 0.993 | 1.0->1.0 | 1.0 | 1.0 |
| servers-os | 3 | 0.993 | 1.0->1.0 | 1.0 | 1.0 |
| security | 1 | - | 1.0->1.0 | 1.0 | 1.0 |
| security | 2 | 1.0 | 1.0->1.0 | 0.95 | 1.0 |
| security | 3 | 0.919 | 1.0->1.0 | 1.0 | 1.0 |
| custom-props | 1 | - | 1.0->1.0 | 0.9 | 0.45 |
| custom-props | 2 | 0.964 | 0.0->1.0 | 0.75 | 0.15 |
| custom-props | 3 | 0.998 | 0.0->0.0 | 0.93 | 1.0 |
| reports | 1 | - | 1.0->1.0 | 1.0 | 1.0 |
| reports | 2 | 0.997 | 1.0->1.0 | 1.0 | 1.0 |
| reports | 3 | 0.992 | 1.0->0.0 | 0.95 | 0.5 |
| avanti | 1 | - | 1.0->1.0 | 1.0 | 1.0 |
| avanti | 2 | 0.756 | 1.0->1.0 | 1.0 | 0.75 |
| avanti | 3 | 0.91 | 0.0->1.0 | 1.0 | 1.0 |
| barcodes | 1 | - | 1.0->1.0 | 1.0 | 1.0 |
| barcodes | 2 | 0.887 | 1.0->1.0 | 1.0 | 1.0 |
| barcodes | 3 | 0.718 | 1.0->1.0 | 0.95 | 0.9 |
| deadline-tracker | 1 | - | 1.0->1.0 | 0.95 | 0.88 |
| deadline-tracker | 2 | 0.992 | 0.0->1.0 | 0.9 | 1.0 |
| deadline-tracker | 3 | 0.947 | 1.0->0.0 | 0.92 | 1.0 |
