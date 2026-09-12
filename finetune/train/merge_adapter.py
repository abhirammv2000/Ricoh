"""Merge the trained LoRA adapter into a full 16-bit checkpoint for serving.

vLLM 0.29.0 dropped bitsandbytes from its supported quantization methods
(confirmed against the actual installed version's error message, not assumed
from docs describing an older vLLM release), so serving the adapter directly
against the 4-bit quantized base it was trained on is not an option with the
version installed here. Merging into a plain 16-bit checkpoint and serving
that is unsloth's own documented path for exactly this situation, and is
simpler regardless: no LoRA-serving flags, no quantization-method flags, just
a standard HF checkpoint.

    python train/merge_adapter.py
"""

from __future__ import annotations

from pathlib import Path

from unsloth import FastLanguageModel

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = REPO_ROOT / "adapter"
MERGED_DIR = REPO_ROOT / "merged_model"

MAX_SEQ_LENGTH = 8192


def main() -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_DIR),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model.save_pretrained_merged(str(MERGED_DIR), tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {MERGED_DIR}")


if __name__ == "__main__":
    main()
