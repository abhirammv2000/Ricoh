"""QLoRA fine-tune of Llama 3.1 8B Instruct on Citera's synthesizer distillation data.

Runs on the GCP L4 VM (see infra/create_vm.sh), not locally: this needs a real
GPU, and unsloth's 4-bit loading path only makes sense with CUDA available.

Hyperparameters below follow unsloth's own published LoRA hyperparameter
guide for an 8B model, not guessed: rank 32, alpha 64 (2x rank, the guide's
upper recommendation, since this is a narrow single-skill distillation
target rather than broad instruction-following), all seven linear layers,
lr 2e-4, 3 epochs (the guide's ceiling for an instruction dataset this size
before overfitting risk rises), effective batch 16 via grad accumulation.

Max sequence length is 8192, not the usual tutorial default of 2048: this
project's training examples embed full retrieved evidence blocks in the user
turn, and the actual data (data/train_examples.jsonl) runs up to ~5800 tokens.
Unsloth benchmarks put a single L4 comfortably past 20k tokens of context at
this LoRA rank, so 8192 has headroom without approaching the card's limit.

Loss is masked to the assistant turn only (unsloth's train_on_responses_only):
without it the model would spend capacity learning to reproduce retrieved
documentation text verbatim, which is not the skill being distilled.

    python train/finetune_qlora.py
"""

from __future__ import annotations

# unsloth has to be the first HF-ecosystem import in the process, before trl,
# transformers, or datasets. It patches those libraries on import, and doing
# that after they are already imported elsewhere left the tokenizer with an
# eos_token unsloth's chat-template patch never actually reconciled against
# the loaded vocabulary (confirmed against unsloth's own reported fix for
# this exact error).
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import json
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "train_examples.jsonl"
OUTPUT_DIR = REPO_ROOT / "adapter"

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 8192

# Held out purely to watch eval loss during training and catch overfitting on
# this small a dataset. Not the judged benchmark: that comparison happens
# later, in the Ricoh repo, against documents this training data never saw.
N_VAL = 20
SEED = 20260911


def load_examples() -> list[dict]:
    examples = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            examples.append({"messages": ex["messages"]})
    return examples


def main() -> None:
    examples = load_examples()
    print(f"Loaded {len(examples)} examples from {DATA_PATH}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # unsloth auto-selects bf16 on Ada Lovelace (L4)
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    def to_text(batch: dict) -> dict:
        return {
            "text": [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in batch["messages"]
            ]
        }

    dataset = Dataset.from_list(examples).shuffle(seed=SEED)
    dataset = dataset.map(to_text, batched=True)
    split = dataset.train_test_split(test_size=N_VAL, seed=SEED)
    train_ds, val_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            packing=False,  # examples are long single turns; packing gains little here
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_ratio=0.1,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=SEED,
            output_dir=str(REPO_ROOT / "checkpoints"),
            report_to="none",
        ),
    )

    # Mask the loss to the assistant turn only. Llama 3.1's chat template
    # wraps each turn in these header tokens; get_chat_template above set the
    # template, this tells the trainer where the boundary between them falls.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    stats = trainer.train()
    print(f"Training complete. {stats}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved LoRA adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
