"""
Step 2: Document ingestion + chunking.

What this does:
  1. Reads every PDF in data/raw/<lang>/  (you organize source docs by language
     folder — see expected layout below).
  2. Extracts text per page using PyMuPDF (fast, good for born-digital PDFs).
     Falls back to pdfplumber if PyMuPDF gets a suspiciously short result
     (common with scanned/odd-encoding PDFs — pdfplumber's layout-aware
     extraction sometimes does better).
  3. Splits text into overlapping chunks (character-based, not token-based —
     simple and good enough for our chunk sizes; token-based chunking is a
     nice-to-have upgrade later, not needed for a first working version).
  4. Writes all chunks to data/processed/chunks.jsonl — one JSON object per
     chunk, with metadata (source file, language, page, chunk index).

Expected input layout (create these folders and drop PDFs in):
  data/raw/en/*.pdf
  data/raw/hi/*.pdf
  data/raw/kn/*.pdf

  (Kanglish isn't a source-document language — it only appears in the test
  QUERIES, not the knowledge base. Skip it here.)

Usage:
  python src/ingest.py
  python src/ingest.py --lang en          # ingest just one language
  python src/ingest.py --min-chars 40     # skip near-empty chunks (headers/footers)

Output:
  data/processed/chunks.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from tqdm import tqdm

# Allow running as `python src/ingest.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS


def extract_text_pymupdf(pdf_path: Path) -> list[str]:
    """Return a list of page texts using PyMuPDF."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text())
    return pages


def extract_text_pdfplumber(pdf_path: Path) -> list[str]:
    """Fallback extractor — slower, sometimes better on tricky layouts/tables."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def extract_pdf_pages(pdf_path: Path, min_chars_per_page: int = 20) -> list[str]:
    """
    Extract text page-by-page, preferring PyMuPDF and falling back to
    pdfplumber only if PyMuPDF's output looks too sparse to be right.
    """
    try:
        pages = extract_text_pymupdf(pdf_path)
    except Exception as e:
        print(f"  [warn] PyMuPDF failed on {pdf_path.name}: {e}. Trying pdfplumber.")
        pages = []

    avg_len = sum(len(p) for p in pages) / max(len(pages), 1)
    if avg_len < min_chars_per_page:
        print(f"  [info] Sparse extraction from PyMuPDF on {pdf_path.name}, trying pdfplumber fallback.")
        try:
            pages = extract_text_pdfplumber(pdf_path)
        except Exception as e:
            print(f"  [warn] pdfplumber also failed on {pdf_path.name}: {e}")

    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """
    Simple sliding-window character chunking with overlap.
    Splits on whitespace boundaries where possible so we don't cut words in half.
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # try to break at the last space before `end` to avoid mid-word cuts
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

    pdf_files = sorted(lang_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[skip] No PDFs found in {lang_dir}")
        return []

    records = []
    for pdf_path in tqdm(pdf_files, desc=f"Ingesting [{lang}]"):
        pages = extract_pdf_pages(pdf_path)
        for page_num, page_text in enumerate(pages, start=1):
            chunks = chunk_text(page_text)
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk) < min_chars:
                    continue
                records.append({
                    "id": f"{lang}__{pdf_path.stem}__p{page_num}__c{chunk_idx}",
                    "text": chunk,
                    "language": lang,
                    "source_file": pdf_path.name,
                    "page": page_num,
                    "chunk_index": chunk_idx,
                })
    return records


def main():
    parser = argparse.ArgumentParser(description="Ingest and chunk source PDFs.")
    parser.add_argument("--lang", choices=["en", "hi", "kn"], default=None,
                         help="Ingest only this language (default: all).")
    parser.add_argument("--min-chars", type=int, default=40,
                         help="Drop chunks shorter than this (filters headers/footers).")
    args = parser.parse_args()

    languages = [args.lang] if args.lang else ["en", "hi", "kn"]

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
        print(f"  {lang}: {n} chunks")


if __name__ == "__main__":
    main()
