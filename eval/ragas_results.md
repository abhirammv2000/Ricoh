# RAGAS cross-check: faithfulness vs our groundedness judge

RAGAS 0.2.14 faithfulness on `claude-sonnet-4-6` against the harness's `claude-opus-5` groundedness score, on 12 of 12 questions sampled from `ragas_input.jsonl`.

This is a consistency probe, not validation: RAGAS decomposes the answer
into claims and checks each against the contexts, with its own prompts, so
agreement is evidence our judge is not idiosyncratic. It is still an
LLM grading an LLM, so it cannot catch a bias the models share. Only the
human labels in `label_for_kappa.py` close that gap.

| | mean |
|---|---|
| RAGAS faithfulness | 0.900 |
| Our groundedness judge | 0.959 |

- Pearson r: 0.371
- Binary agreement at >= 0.8: 75% raw, chance 78%, Cohen's kappa -0.125

With n=12 and both raters scoring almost everything above the threshold,
chance agreement is near the raw number, so the kappa carries little
information either way. Read the means and the disagreements, not the kappa.

## Disagreements (3)

| Q | RAGAS faithfulness | our groundedness | question |
|---|---|---|---|
| 71 | 0.727 | 1.0 | When using a PCLOut printer, does it save AFP resources like fonts or  |
| 86 | 0.8 | 0.75 | When multiple text edit options are applied together in the AFP indexi |
| 26 | 0.778 | 0.95 | When sorting an AFP job by customer last name, does RICOH ProcessDirec |
