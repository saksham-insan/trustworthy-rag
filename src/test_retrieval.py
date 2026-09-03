"""
Step 3b: Test retrieval — query both index conditions and see them in action.

What this does:
  Takes a sample question, embeds it (with the e5 "query: " prefix — the
  counterpart to the "passage: " prefix used when we indexed chunks), and
  retrieves the top-k most similar chunks from:
    (a) the MONOLINGUAL index for the question's language only
    (b) the MULTILINGUAL shared index (which could return chunks from
        ANY language, even if the question is in English)

  This lets you visually inspect: does the multilingual index ever pull in
  a chunk from the wrong language? Does retrieval quality look reasonable
  at all? This is a sanity check before we wire retrieval into the full
  generator + verifier pipeline.

Usage:
  python src/test_retrieval.py "What is the eligibility for PM-USP scholarship?" en
  python src/test_retrieval.py "पात्रता क्या है?" hi
"""

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CHROMA_MONO_DIR, CHROMA_MULTI_DIR, EMBEDDING_MODEL_NAME, TOP_K


def embed_query(model: SentenceTransformer, query: str):
    # e5 convention: queries get "query: ", passages got "passage: " at index time
    return model.encode([f"query: {query}"], convert_to_numpy=True).tolist()


def print_results(label: str, results):
    print(f"\n--- {label} ---")
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    if not docs:
        print("  (no results)")
        return
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        preview = doc[:150].replace("\n", " ")
        print(f"  {i}. [{meta['language']} | {meta['scheme_slug']}] (distance={dist:.4f})")
        print(f"     {preview}...")


def main():
    if len(sys.argv) < 3:
        print('Usage: python src/test_retrieval.py "your question" <lang: en|hi|bn>')
        sys.exit(1)

    query = sys.argv[1]
    lang = sys.argv[2]
    if lang not in ("en", "hi", "bn"):
        print("Language must be one of: en, hi, bn")
        sys.exit(1)

    print(f"Query: {query!r}  (language: {lang})")
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    query_embedding = embed_query(model, query)

    # (a) Monolingual condition — search only within this language's collection
    mono_client = chromadb.PersistentClient(path=str(CHROMA_MONO_DIR))
    mono_collection = mono_client.get_collection(name=f"lang_{lang}")
    mono_results = mono_collection.query(query_embeddings=query_embedding, n_results=TOP_K)
    print_results(f"MONOLINGUAL index (lang_{lang} only)", mono_results)

    # (b) Multilingual condition — search the shared collection, any language can surface
    multi_client = chromadb.PersistentClient(path=str(CHROMA_MULTI_DIR))
    multi_collection = multi_client.get_collection(name="all_languages")
    multi_results = multi_collection.query(query_embeddings=query_embedding, n_results=TOP_K)
    print_results("MULTILINGUAL index (all languages combined)", multi_results)


if __name__ == "__main__":
    main()