"""
Step 2: Document ingestion + chunking.

What this does:
  1. Reads every .txt file in data/raw/<lang>/  (you organize source docs by
     language folder — see expected layout below). Each .txt file holds the
     copy-pasted webpage content for one scheme (Details, Benefits,
     Eligibility, etc. — myScheme doesn't offer PDF downloads, so we work
     with the page text directly).
  2. Cleans up stray formatting characters left over from copy-pasting off
     the web (e.g. the invisible BOM/zero-width character U+FEFF).
  3. Splits text into overlapping chunks (character-based — simple and good
     enough for our chunk sizes; token-based chunking is a nice upgrade
     later, not needed for a first working version).
  4. Writes all chunks to data/processed/chunks.jsonl — one JSON object per
     chunk, with metadata (source file, language, chunk index).

Expected input layout:
  data/raw/en/<scheme-slug>.txt
  data/raw/hi/<scheme-slug>.txt
  data/raw/bn/<scheme-slug>.txt

  The SAME slug (filename) must be reused across all three language
  folders for the same scheme — that's what lets later analysis match
  "this English chunk" to "this Hindi/Bengali chunk" of the same content.

  (Benglish/code-mixed text isn't a source-document language — it only
  appears in the test QUERIES later, not the knowledge base. Skip it here.)

Usage:
  python src/ingest.py
  python src/ingest.py --lang en          # ingest just one language
  python src/ingest.py --min-chars 40     # skip near-empty chunks

Output:
  data/processed/chunks.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Allow running as `python src/ingest.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS, SUPPORTED_LANGUAGES


def clean_text(text: str) -> str:
    """Strip stray characters commonly left behind by copy-pasting from web pages."""
    text = text.replace("\ufeff", " ")  # zero-width no-break space / BOM
    text = text.replace("\u200b", " ")  # zero-width space
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """
    Simple sliding-window character chunking with overlap.
    Splits on whitespace boundaries where possible so we don't cut words in half.
    """
    text = " ".join(text.split())  # normalize whitespace (also collapses blank lines)
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end - overlap > start else end  # guard against infinite loop
    return chunks


def ingest_language(lang: str, min_chars: int) -> list[dict]:
    lang_dir = RAW_DATA_DIR / lang
    if not lang_dir.exists():
        print(f"[skip] No folder found for language '{lang}' at {lang_dir}")
        return []

    txt_files = sorted(lang_dir.glob("*.txt"))
    if not txt_files:
        print(f"[skip] No .txt files found in {lang_dir}")
        return []

    records = []
    for txt_path in tqdm(txt_files, desc=f"Ingesting [{lang}]"):
        raw_text = txt_path.read_text(encoding="utf-8")
        text = clean_text(raw_text)
        chunks = chunk_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk) < min_chars:
                continue
            records.append({
                "id": f"{lang}__{txt_path.stem}__c{chunk_idx}",
                "text": chunk,
                "language": lang,
                "scheme_slug": txt_path.stem,   # e.g. "pmusp" — same across languages
                "source_file": txt_path.name,
                "chunk_index": chunk_idx,
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="Ingest and chunk source .txt files.")
    parser.add_argument("--lang", choices=["en", "hi", "bn"], default=None,
                         help="Ingest only this language (default: all).")
    parser.add_argument("--min-chars", type=int, default=40,
                         help="Drop chunks shorter than this.")
    args = parser.parse_args()

    languages = [args.lang] if args.lang else ["en", "hi", "bn"]

    all_records = []
    for lang in languages:
        all_records.extend(ingest_language(lang, args.min_chars))

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(all_records)} chunks to {out_path}")
    for lang in languages:
        n = sum(1 for r in all_records if r["language"] == lang)
        schemes = sorted(set(r["scheme_slug"] for r in all_records if r["language"] == lang))
        print(f"  {lang}: {n} chunks across {len(schemes)} schemes {schemes}")


if __name__ == "__main__":
    main()