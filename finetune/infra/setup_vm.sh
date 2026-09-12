#!/bin/bash
# Run on the GCP VM itself (via gcloud compute ssh) to install unsloth.
#
# Uses unsloth's own auto-install detection script rather than a hand-picked
# pip command with version pins: the DLVM image ships a specific PyTorch/CUDA
# combination (2.9 / cu129), and the auto-installer inspects the actual
# installed torch build and CUDA version on THIS machine to produce a command
# guaranteed to match it, instead of guessing pins that could be stale by the
# time this script runs.

set -euo pipefail

# The DLVM image's system Python is externally managed (PEP 668, Ubuntu
# 24.04) and torch is already correctly installed into it, wired up against
# the image's pre-baked CUDA 12.9 stack. A fresh venv would mean reinstalling
# torch and re-verifying that wiring for no benefit on a single-purpose,
# short-lived training VM, so --break-system-packages (pip's own documented
# override for exactly this case) is used instead of a venv.
python3 -m pip install --upgrade pip --break-system-packages

# unsloth's own auto-install detector (_auto_install.py) hardcodes a CUDA
# version whitelist (11.8/12.1/12.4/12.6/12.8/13.0) that does not yet include
# 12.9, which is what this DLVM image's torch build reports, so the detector
# raises rather than producing a command. That is a gap in the detector, not
# a real incompatibility: torch/CUDA 12.9 already imports and sees the GPU
# fine (verified separately). Falling back to unsloth's other officially
# documented install path, plain pip install, which lets pip's resolver pick
# compatible dependency versions against the torch already installed here,
# rather than force-installing wheels pinned to a different CUDA line.
python3 -m pip install unsloth trl datasets --break-system-packages

python3 -c "import torch, unsloth; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available()); print('unsloth import OK')"
