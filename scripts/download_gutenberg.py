"""Download the public-domain books listed in data/sources.yaml from Project Gutenberg.

Usage:
    python scripts/download_gutenberg.py
"""
import re
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "public_domain"
GUTENDEX = "https://gutendex.com/books"

START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)


def find_book(title: str, author: str) -> dict | None:
    resp = requests.get(GUTENDEX, params={"search": f"{title} {author}".strip()}, timeout=30)
    resp.raise_for_status()
    for book in resp.json().get("results", []):
        authors = " ".join(a["name"] for a in book.get("authors", []))
        if author.lower() in authors.lower() and book.get("copyright") is False:
            return book
    return None


def plain_text_url(book: dict) -> str | None:
    for mime, url in book.get("formats", {}).items():
        if mime.startswith("text/plain") and not url.endswith(".zip"):
            return url
    return None


def strip_gutenberg_boilerplate(text: str) -> str:
    start = START_RE.search(text)
    if start:
        text = text[start.end():]
    end = END_RE.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = yaml.safe_load((ROOT / "data" / "sources.yaml").read_text())["books"]

    for entry in sources:
        title, author = entry["title"], entry.get("author", "")
        book = find_book(title, author)
        url = book and plain_text_url(book)
        if not url:
            print(f"[skip] not found: {title} ({author or 'anon'})")
            continue

        out = OUT_DIR / f"pg{book['id']}.txt"
        if out.exists():
            print(f"[have] {book['title']}")
            continue

        text = requests.get(url, timeout=60).text
        out.write_text(strip_gutenberg_boilerplate(text), encoding="utf-8")
        print(f"[ok]   {book['title']} -> {out.name}")
        time.sleep(1)  # be polite to Gutenberg


if __name__ == "__main__":
    main()
