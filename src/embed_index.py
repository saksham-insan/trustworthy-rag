"""
Step 3: Embed chunks + build the two index conditions.

What this does:
  1. Loads data/processed/chunks.jsonl (output of ingest.py).
  2. Embeds every chunk using multilingual-e5-base (runs on CPU, batched to
     stay light on 8GB RAM).
  3. Builds TWO separate ChromaDB stores — this is the core of your
     research contribution (RQ3):
       (a) MONOLINGUAL condition: one ChromaDB collection PER language
           (en, hi, bn) — languages never mix in retrieval.
       (b) MULTILINGUAL condition: ONE ChromaDB collection with all
           languages combined — a query can retrieve chunks from any
           language.
  Both conditions are built from the exact same underlying chunks, so any
  difference in retrieval accuracy between them is attributable to the
  indexing condition itself, not to different data.

IMPORTANT — e5 model convention: multilingual-e5-base was trained expecting
a "passage: " prefix on documents being indexed, and a "query: " prefix on
search queries at retrieval time. This isn't optional styling — skipping it
measurably hurts retrieval quality. We handle the "passage: " side here;
the "query: " side happens later in the retrieval script.

Usage:
  python src/embed_index.py

Output:
  chroma_db_mono/   (3 collections: en, hi, bn)
  chroma_db_multi/  (1 collection: all_languages)
"""

import truststore
truststore.inject_into_ssl()  # must run before any network calls (model download)

import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    PROCESSED_DATA_DIR,
    CHROMA_MONO_DIR,
    CHROMA_MULTI_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
)


def load_chunks() -> list[dict]:
    chunks_path = PROCESSED_DATA_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"{chunks_path} not found. Run `python src/ingest.py` first."
        )
    records = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """
    Embed a list of chunk texts with the e5 'passage: ' prefix, in small
    batches to stay light on RAM.
    """
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def build_monolingual_indexes(records: list[dict], model: SentenceTransformer):
    print("\n=== Building MONOLINGUAL indexes (one collection per language) ===")
    client = chromadb.PersistentClient(path=str(CHROMA_MONO_DIR))

    languages = sorted(set(r["language"] for r in records))
    for lang in languages:
        lang_records = [r for r in records if r["language"] == lang]
        texts = [r["text"] for r in lang_records]

        print(f"\n[{lang}] Embedding {len(texts)} chunks...")
        embeddings = embed_texts(model, texts)

        # Fresh collection each run so re-running this script doesn't duplicate data
        # (ChromaDB collection names must be 3+ characters — "bn"/"en"/"hi" alone
        # are too short, so we prefix them)
        collection_name = f"lang_{lang}"
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass  # collection didn't exist yet, fine
        collection = client.create_collection(name=collection_name)

        collection.add(
            ids=[r["id"] for r in lang_records],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"language": r["language"], "scheme_slug": r["scheme_slug"], "source_file": r["source_file"]}
                for r in lang_records
            ],
        )
        print(f"[{lang}] Indexed {collection.count()} chunks into chroma_db_mono/{collection_name}")


def build_multilingual_index(records: list[dict], model: SentenceTransformer):
    print("\n=== Building MULTILINGUAL index (one shared collection, all languages) ===")
    client = chromadb.PersistentClient(path=str(CHROMA_MULTI_DIR))

    texts = [r["text"] for r in records]
    print(f"Embedding {len(texts)} chunks (all languages combined)...")
    embeddings = embed_texts(model, texts)

    try:
        client.delete_collection(name="all_languages")
    except Exception:
        pass
    collection = client.create_collection(name="all_languages")

    collection.add(
        ids=[r["id"] for r in records],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"language": r["language"], "scheme_slug": r["scheme_slug"], "source_file": r["source_file"]}
            for r in records
        ],
    )
    print(f"Indexed {collection.count()} chunks into chroma_db_multi/all_languages")


def main():
    records = load_chunks()
    print(f"Loaded {len(records)} chunks from chunks.jsonl")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")
    print("(First run downloads ~1.1GB — may take a few minutes depending on connection.)")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    build_monolingual_indexes(records, model)
    build_multilingual_index(records, model)

    print("\nDone. Both index conditions built successfully.")


if __name__ == "__main__":
    main()