"""Turn raw .txt books into a JSONL training set of overlapping prose chunks.

Reads every .txt under data/raw/public_domain and data/raw/licensed,
cleans it, splits it into ~N-word passages on paragraph boundaries and
writes data/processed/train.jsonl and data/processed/val.jsonl.

Usage:
    python scripts/prepare_dataset.py --chunk-words 700
"""
import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIRS = [ROOT / "data" / "raw" / "public_domain", ROOT / "data" / "raw" / "licensed"]
OUT_DIR = ROOT / "data" / "processed"


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Re-flow hard-wrapped lines inside paragraphs.
    paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paragraphs if p)


def chunk(paragraphs: list[str], chunk_words: int) -> list[str]:
    chunks, current, count = [], [], 0
    for para in paragraphs:
        words = len(para.split())
        if count + words > chunk_words and current:
            chunks.append("\n\n".join(current))
            # Keep the last paragraph so consecutive chunks overlap a little.
            current, count = current[-1:], len(current[-1].split())
        current.append(para)
        count += words
    if count > chunk_words // 4:
        chunks.append("\n\n".join(current))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-words", type=int, default=700)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = []
    for raw_dir in RAW_DIRS:
        for path in sorted(raw_dir.glob("*.txt")):
            paragraphs = clean(path.read_text(encoding="utf-8", errors="ignore")).split("\n\n")
            pieces = chunk(paragraphs, args.chunk_words)
            samples += [{"text": p, "source": path.name} for p in pieces]
            print(f"{path.name}: {len(pieces)} chunks")

    if not samples:
        raise SystemExit("No .txt files found. Run scripts/download_gutenberg.py first.")

    random.Random(args.seed).shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_fraction))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("val", samples[:n_val]), ("train", samples[n_val:])):
        with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"train={len(samples) - n_val} val={n_val} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
