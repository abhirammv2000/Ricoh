# Fine-tuning the synthesizer

A QLoRA fine-tune of Llama 3.1 8B Instruct that distills Citera's synthesizer
skill: given retrieved evidence and a question, write a grounded, cited
answer, or refuse in the exact machine-readable way when the evidence doesn't
cover it. Self-hosted on a single GCP L4 GPU behind vLLM, then judged
head-to-head against the existing Claude/GPT-4o-mini/Gemini bakeoff on the
same 20 questions.

**Result:** correctness 0.941 against Claude Sonnet's 0.9925, inside the
judge's ~0.10 noise floor. Groundedness (0.9055 vs. 0.98) is a real, smaller
gap. Latency is 25.2s vs. 2.6-9.9s for the commercial models, expected for one
unbatched L4 versus provider-scale serving infrastructure. Full table in
[`../eval/provider_bakeoff.md`](../eval/provider_bakeoff.md), which is where
the actual comparison runs (this folder trains and serves the model; the
judged eval reuses Citera's own harness and judge rather than duplicating
it).

## Why this exists

RAG, agentic orchestration, and an LLM-as-judge eval harness are what Citera
covers. Fine-tuning and self-hosted serving are a different, complementary
skill: can a small open-weight model, trained on distilled examples of one
specific behavior, close the gap with a commercial API for that behavior at a
fraction of the per-token cost (self-hosted serving is billed as GPU-hours,
not per-token, and the two are not directly comparable, but the direction of
the tradeoff is the point).

## Pipeline

```
Ricoh's 733 docs                              Citera's real retriever
  |                                                    |
  v                                                    v
593 disjoint documents  ---(sample chunk)-->  question  ---(retrieve)-->  evidence
(never touched by any                            |                          |
 Citera eval set)                                v                          v
                                          Claude Sonnet (teacher)  <---------+
                                                  |
                                                  v
                                    348 (question, evidence, answer) pairs
                                                  |
                                                  v
                                    QLoRA fine-tune, Llama 3.1 8B Instruct
                                                  |
                                                  v
                                    merge to 16-bit  -->  vLLM  -->  judged
                                                                     against
                                                                     Citera's
                                                                     bakeoff
```

### 1. Leakage-safe training data

Citera's eval suite (the 100-question benchmark, ground truth, multi-hop and
multi-turn sets) draws on 140 of the corpus's 733 documents. `data/`
computes the other 593 and samples training questions only from those, so
training a model and then judging it on Citera's existing benchmark is a
genuine held-out test, not train/eval overlap, by construction (different
source documents entirely, not just different questions).

Each training example is built the way a real synthesizer call is built in
production, not synthesized separately: `scripts/generate_training_data.py`
samples a chunk, asks a generator model for a technician-style question
answerable from it (same generator prompt Citera's own
`eval/generate_questions.py` uses), runs that question through Citera's real
hybrid retriever, then has Claude Sonnet answer with Citera's actual
`SYNTHESIZER_PROMPT` against whatever evidence retrieval actually returned.
That last part matters: 11 of the 348 examples (3.2%) are honest refusals,
because retrieval didn't always find the source document, and the model is
supposed to learn "refuse when evidence doesn't support an answer" as much as
"cite correctly when it does."

348 examples, 2 rejected by the generator as unanswerable-from-that-excerpt,
0 failed API calls in the end. Cost: **~$7** (a 3-example pilot at $0.0527,
then a 350-example run that crashed after 29 examples with no traceback at
all, likely a transient native-library or network issue and not a bug, since
a plain 15-minute background sleep probe confirmed the harness itself has no
hidden timeout that would explain it, resumed from item 30 for $6.4732; the
first partial run's exact cost was never logged, since the manifest write
happens at the end and the crash preempted it, so the total is an estimate
built from the other two runs' per-example rate).

### 2. QLoRA fine-tune

`train/finetune_qlora.py`, on a GCP `g2-standard-8` (1x L4, 24GB) via
unsloth. Hyperparameters from unsloth's own published LoRA guide, not
guessed: rank 32, alpha 64 (2x rank, the guide's upper end, since this is a
narrow single-skill distillation target rather than broad instruction
tuning), all seven linear layers, lr 2e-4, 3 epochs, effective batch 16.

Max sequence length is 8192, not the usual tutorial default of 2048: these
training examples embed full retrieved evidence blocks in the user turn and
run up to ~5,800 tokens. unsloth benchmarks put a single L4 past 20k tokens
of context at this rank, so 8192 has headroom without touching the card's
limit.

328 train / 20 held-out val (watching eval loss only, not the judged
benchmark, which never touches training documents at all). 3 epochs, 63
steps, ~65 minutes:

| epoch | eval loss |
|---|---|
| 0.49 | 0.568 |
| 0.98 | 0.510 |
| 1.44 | 0.502 |
| 1.93 | **0.483** (minimum) |
| 2.39 | 0.528 |
| 2.88 | 0.516 |
| 3.00 | 0.516 |

Eval loss bottoms out around epoch 1.9 and creeps back up slightly rather
than continuing to fall, a mild overfitting signal on a dataset this small.
It plateaus rather than climbing further, so the final checkpoint (epoch 3,
what's actually shipped, since `load_best_model_at_end` wasn't configured)
is a little past its best point, not badly overfit. Worth knowing, not worth
redoing training over a 0.033 loss difference.

### 3. Serving

vLLM 0.29.0, in its own venv (isolated from the training environment's torch
build the same way Citera's own `.venv-ragas` is isolated from its main
`.venv`, for the same reason: two dependency trees that don't agree on a
torch version). Two things broke here that are worth knowing about if you're
doing this in 2026:

- **vLLM 0.29.0 dropped `bitsandbytes` from its supported quantization
  methods entirely** (confirmed against the actual installed version's error
  message, which enumerates the supported list). Serving the LoRA adapter
  directly against the 4-bit base it trained on, the "obvious" choice since
  it matches training precision exactly, is not an option with this vLLM
  version. `train/merge_adapter.py` merges the adapter into a plain 16-bit
  checkpoint instead, unsloth's own documented path for exactly this
  situation, and `serve/serve_vllm.sh` serves that: no LoRA flags, no
  quantization flags, a standard checkpoint.
- **vLLM shells out to `ninja` by bare name** to compile CUDA kernels at
  startup. Running it by full path from an unactivated venv means `ninja`
  isn't on `PATH` and the engine fails to start with no obvious hint that
  the venv itself is the problem. Activate the venv, don't just invoke its
  binary.

The endpoint is reached over an SSH tunnel from the evaluating machine, not
a public firewall rule: vLLM's OpenAI-compatible server has no built-in
auth, so opening the port to `0.0.0.0/0` would put an unauthenticated LLM
endpoint on the open internet for a few dollars of eval convenience.

### 4. Wired into Citera

`../src/llm_factory.py` gained a fourth provider, `self_hosted`, following
the exact pattern already used for the OpenAI and Gemini providers added for
the cross-provider bakeoff: a `ChatOpenAI` client pointed at a different base
URL, gated behind `SELF_HOSTED_LLM_BASE_URL`. `../eval/provider_bakeoff.py`
then runs this model through the same harness, same shared retrieval, same
fixed Opus judge as the other three providers, on the same 20 questions.

## Reproducing

Run from the Ricoh repo root, with its retrieval index already built: this
reuses Citera's real retriever and `SYNTHESIZER_PROMPT` directly rather than
a copy that could drift.

```bash
# 1. Generate training data
python -m finetune.scripts.generate_training_data --pilot   # 3 examples, ~$0.10, sanity check first
python -m finetune.scripts.generate_training_data --n 350

# 2. On a GPU VM (finetune/infra/create_vm.sh for the exact GCP setup used).
# Only finetune/train/ and finetune/data/train_examples.jsonl need to be on
# the VM, not the whole repo.
bash finetune/infra/setup_vm.sh
python3 finetune_qlora.py
python3 merge_adapter.py

# 3. Serve
bash finetune/serve/serve_vllm.sh

# 4. Back on the machine running evals, add SELF_HOSTED_LLM_BASE_URL to .env:
python -m eval.provider_bakeoff --providers self_hosted --n 20
```

## What this doesn't show

- **One judged run, 20 questions.** Citera's own eval work is built around
  the finding that a 10-question benchmark gives flattering, noisy numbers;
  20 is better than 10 but still narrow. The honest fix is the same one
  Citera already applied to itself: run this against the full 100-question
  benchmark's holdout split before trusting the numbers past "the direction
  looks right."
- **No DPO/preference tuning.** This is supervised fine-tuning on teacher
  demonstrations only. Whether preference optimization narrows the
  groundedness gap further is untested.
- **Single training run, no hyperparameter sweep.** The rank/alpha/epoch
  choices came from unsloth's published guide, not from comparing
  alternatives on this specific dataset.
- **Cost is not apples-to-apples.** $0/query for a metered, always-on GPU
  server versus per-token API pricing are different cost models; the
  provider_bakeoff table's `self_hosted` row says $0.00000 for exactly this
  reason and that number should not be read as "free."

## Cost

| step | cost |
|---|---|
| training data generation (348 examples + pilot) | ~$7.00 |
| GCP `g2-standard-8` + 1x L4, ~5.3 hrs (training, serving, all the debugging) | ~$4.55 |
| bakeoff judge (Opus, 20 questions) | $0.55 |
| **total** | **~$12.15** |
