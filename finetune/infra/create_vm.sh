#!/bin/bash
# Create the single GPU VM used for both training and serving.
#
# One L4 is all the project has quota for (verified: NVIDIA_L4_GPUS limit 1 in
# us-central1), so training and serving share this one instance rather than
# provisioning two. On-demand, not spot: training is a single ~30-60 minute
# run and a preemption mid-run would cost more in lost setup time than the
# ~$0.68/hr on-demand premium over spot saves.
#
# Image, zone, and machine type below were verified live against this GCP
# project before writing this script (gcloud compute images list / gcloud
# compute accelerator-types list), not assumed from documentation that could
# be stale:
#   - pytorch-2-9-cu129-ubuntu-2404-nvidia-580: current DLVM image family,
#     ships PyTorch 2.9, CUDA 12.9, driver 580 preinstalled. No manual CUDA
#     or driver setup needed.
#   - us-central1-a: confirmed to actually offer nvidia-l4 accelerators.
#   - g2-standard-8: smallest G2 shape (8 vCPU, 32GB RAM), 1x L4 GPU.
#
# Cost: ~$0.85/hr on-demand (checked live pricing at write time). A training
# run this small should finish in well under two hours end to end including
# setup.

set -euo pipefail

PROJECT="final-project-478101"
ZONE="us-central1-a"
INSTANCE="citera-finetune-l4"

gcloud compute instances create "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=g2-standard-8 \
  --accelerator="type=nvidia-l4,count=1" \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=150GB \
  --boot-disk-type=pd-ssd

echo "Instance created. Stop it the moment training + serving are done:"
echo "  gcloud compute instances stop $INSTANCE --project=$PROJECT --zone=$ZONE"
echo "Delete it once the fine-tuning work is fully wrapped up:"
echo "  gcloud compute instances delete $INSTANCE --project=$PROJECT --zone=$ZONE"
