# Multi-turn follow-up evaluation

**Generated:** 2026-09-10 16:06:05  
**Judge:** `claude-opus-5`  
**Turns:** 36 across 12 chains  

`standalone_target` is the hand-written ideal rewrite. Follow-up turns carry the real prior answers as history.

## Condensation

- Mean cosine(rewrite, target) on follow-ups: **0.933**
- Retriever recall on the raw follow-up: 0.562
- Retriever recall after condensation: **0.792**

The recall lift is the point: a follow-up like "what about those?" retrieves nothing on its own; the rewrite makes it a real query.

## Answer quality, first turn vs follow-ups

| | first turns | follow-ups |
|---|---|---|
| n | 12 | 24 |
| evidence recall | 0.875 | 0.792 |
| behaviour match | 1.0 | 0.958 |
| groundedness | 0.949 | 0.947 |
| correctness | 0.892 | 0.823 |

Agent cost $0.64199, judge overhead $1.03313.

## Per-turn

| chain | turn | rewrite~target | recall raw->condensed | grounded | correct |
|---|---|---|---|---|---|
| workflow | 1 | - | 1.0->1.0 | 0.85 | 0.3 |
| workflow | 2 | 0.994 | 0.0->0.0 | 0.95 | 0.0 |
| workflow | 3 | 0.991 | 0.0->1.0 | 0.9 | 0.15 |
| shutdown | 1 | - | 1.0->1.0 | 1.0 | 1.0 |
| shutdown | 2 | 0.988 | 1.0->1.0 | 1.0 | 1.0 |
| shutdown | 3 | 0.956 | 0.5->1.0 | 0.9 | 1.0 |
| fusionpro | 1 | - | 0.5->0.5 | 1.0 | 1.0 |
| fusionpro | 2 | 1.0 | 0.0->1.0 | 1.0 | 1.0 |
| fusionpro | 3 | 0.72 | 0.0->1.0 | 1.0 | 1.0 |
| pclout | 1 | - | 1.0->1.0 | 0.95 | 1.0 |
| pclout | 2 | 0.854 | 1.0->1.0 | 1.0 | 1.0 |
| pclout | 3 | 0.97 | 0.0->1.0 | 0.9 | 1.0 |
| inserters | 1 | - | 1.0->1.0 | 0.85 | 0.9 |
| inserters | 2 | 0.943 | 1.0->1.0 | 1.0 | 1.0 |
| inserters | 3 | 0.678 | 0.0->1.0 | 0.95 | 1.0 |
| servers-os | 1 | - | 1.0->1.0 | 1.0 | 1.0 |
| servers-os | 2 | 0.993 | 1.0->1.0 | 1.0 | 1.0 |
| servers-os | 3 | 0.963 | 1.0->1.0 | 1.0 | 1.0 |
| security | 1 | - | 1.0->1.0 | 0.97 | 1.0 |
| security | 2 | 1.0 | 1.0->1.0 | 0.95 | 1.0 |
| security | 3 | 0.919 | 1.0->1.0 | 0.97 | 1.0 |
| custom-props | 1 | - | 0.0->0.0 | 0.92 | 0.65 |
| custom-props | 2 | 0.964 | 0.0->0.0 | 0.75 | 0.2 |
| custom-props | 3 | 0.998 | 0.0->0.0 | 0.9 | 1.0 |
| reports | 1 | - | 1.0->1.0 | 1.0 | 1.0 |
| reports | 2 | 0.997 | 1.0->1.0 | 1.0 | 1.0 |
| reports | 3 | 0.992 | 1.0->0.0 | 0.97 | 0.45 |
| avanti | 1 | - | 1.0->1.0 | 1.0 | 1.0 |
| avanti | 2 | 0.784 | 1.0->1.0 | 0.8 | 0.0 |
| avanti | 3 | 0.88 | 0.0->1.0 | 1.0 | 1.0 |
| barcodes | 1 | - | 1.0->1.0 | 1.0 | 1.0 |
| barcodes | 2 | 0.887 | 1.0->1.0 | 1.0 | 1.0 |
| barcodes | 3 | 0.979 | 1.0->1.0 | 0.95 | 1.0 |
| deadline-tracker | 1 | - | 1.0->1.0 | 0.85 | 0.85 |
| deadline-tracker | 2 | 0.992 | 0.0->1.0 | 0.92 | 1.0 |
| deadline-tracker | 3 | 0.947 | 1.0->0.0 | 0.92 | 0.95 |
