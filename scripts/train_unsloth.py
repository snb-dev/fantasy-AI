"""Fine-tune Fantasy AI on a small GPU (e.g. an 8 GB laptop) with Unsloth.

Unsloth patches the model for faster, lower-memory QLoRA training, so a
3B-4B model trains comfortably in 8 GB of VRAM and a 7B-8B model just fits
at a short sequence length.

Usage (8 GB VRAM):
    python scripts/train_unsloth.py                                   # Qwen2.5-3B, safe default
    python scripts/train_unsloth.py --base-model unsloth/Meta-Llama-3.1-8B-bnb-4bit \
        --max-seq-length 1024 --batch-size 1 --grad-accum 16           # tighter fit, better prose

If you hit "CUDA out of memory": lower --max-seq-length (1024 -> 768 -> 512),
keep --batch-size 1, and close anything else using the GPU (browsers, games).
"""
# Unsloth must be imported before transformers/trl so its patches apply.
from unsloth import FastLanguageModel, is_bfloat16_supported  # isort: skip

import argparse
from pathlib import Path

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-3B-bnb-4bit")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "fantasy-ai-lora"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",  # offloads activations; big VRAM saver
        random_state=3407,
    )

    data = load_dataset(
        "json",
        data_files={
            "train": str(ROOT / "data" / "processed" / "train.jsonl"),
            "validation": str(ROOT / "data" / "processed" / "val.jsonl"),
        },
    )

    # The prose chunks are plain continuation text; end each with EOS so the
    # model learns where a passage stops.
    data = data.map(lambda row: {"text": row["text"] + tokenizer.eos_token})

    config = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        per_device_eval_batch_size=1,
        save_steps=200,
        save_total_limit=3,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
