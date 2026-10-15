"""Fine-tune an open-weights base model on the fantasy corpus with QLoRA.

This teaches an already-fluent model the *voice* of fantasy prose; it does not
train a model from scratch (that would need billions of words and a GPU cluster).

Fits on one 24 GB GPU (RTX 3090/4090, A10, L4) with the default 8B model,
or a free Colab T4 with a smaller model such as Qwen/Qwen2.5-3B.

Usage:
    python scripts/train_lora.py --base-model meta-llama/Llama-3.1-8B --epochs 1
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "fantasy-ai-lora"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    data = load_dataset(
        "json",
        data_files={
            "train": str(ROOT / "data" / "processed" / "train.jsonl"),
            "validation": str(ROOT / "data" / "processed" / "val.jsonl"),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

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
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
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
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
