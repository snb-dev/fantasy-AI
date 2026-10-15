"""Write with Fantasy AI: load the base model + trained LoRA adapter and continue a prompt.

Usage:
    python scripts/generate.py --prompt "The last dragon-speaker of Vael Morrow woke to ash on her tongue."
    python scripts/generate.py --mode screenplay --prompt "EXT. DROWNED CITY OF KESH - NIGHT"
"""
import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent

MODE_PREFIX = {
    "novel": "",
    # The corpus is prose, so a screenplay header nudges the format; for real
    # script quality, add licensed or self-written screenplays to data/raw/licensed.
    "screenplay": "FADE IN:\n\n",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", help="defaults to the model the adapter was trained on")
    parser.add_argument("--adapter", default=str(ROOT / "outputs" / "fantasy-ai-lora"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", choices=MODE_PREFIX, default="novel")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--load-in-4bit", action="store_true", help="fit 7B-8B models in 8 GB VRAM")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base_model = args.base_model or PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    quant = None
    if args.load_in_4bit or "bnb-4bit" in base_model:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, quantization_config=quant, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.adapter)

    text = MODE_PREFIX[args.mode] + args.prompt
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=0.95,
        repetition_penalty=1.1,
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
