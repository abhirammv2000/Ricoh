#!/bin/bash
# Serve the merged model with vLLM's OpenAI-compatible API.
#
# Run this on the GPU VM itself (see ../infra/create_vm.sh), after
# ../train/merge_adapter.py has produced ../merged_model/.
#
# vLLM installs into its own venv, not the training environment: vLLM 0.29.0
# is compiled against a different torch than the unsloth training stack ends
# up with (unsloth's own pip resolution pulled torch 2.12.1+cu130; vLLM wants
# its own compatible build), so the two would conflict sharing one
# environment. This mirrors the .venv / .venv-ragas split already used in the
# main Citera project for the same reason: isolate conflicting dependency
# trees rather than fight them into one environment.
#
#   python3 -m venv ~/vllm-env
#   source ~/vllm-env/bin/activate
#   pip install vllm
#
# The venv has to be actually activated (not just invoked by full path) before
# starting the server: vLLM shells out to `ninja` by bare name to compile CUDA
# kernels at startup, and that only resolves if the venv's bin/ is on PATH.
#
# --quantization/--enable-lora are deliberately absent: vLLM 0.29.0 dropped
# bitsandbytes from its supported quantization methods (confirmed against the
# actual installed version's error message, which enumerates the supported
# list and bitsandbytes is not in it), so serving the adapter directly against
# the 4-bit base it trained on is not an option with this vLLM version. The
# merged 16-bit checkpoint sidesteps that: it is just a standard HF checkpoint,
# no special serving flags needed.

set -euo pipefail

source "$HOME/vllm-env/bin/activate"

cd "$HOME"
vllm serve "$HOME/citera-finetune/merged_model" \
  --served-model-name citera-finetuned \
  --max-model-len 8192 \
  --port 8000

# From your own machine, reach it over an SSH tunnel rather than opening a
# public firewall rule (vLLM's server has no built-in auth, so exposing 8000
# to 0.0.0.0/0 would put an unauthenticated LLM endpoint on the open internet):
#
#   gcloud compute ssh citera-finetune-l4 --zone=us-central1-a -- -L 8000:localhost:8000 -N
#
# Then point Citera's llm_factory.py at it:
#   SELF_HOSTED_LLM_BASE_URL=http://localhost:8000/v1
