"""
Step 6: Full pipeline + SQLite logging.

What this does:
  Ties together retrieval -> generation -> verification into one function,
  timing each stage, and logs EVERYTHING to SQLite: query, language, index
  condition used, retrieved chunks, generated answer, verifier verdict,
  and latency per stage. This is the data source for all your evaluation
  tables/charts later (retrieval precision, hallucination rate, latency
  trade-offs — RQ1-RQ4).

  Supports both index conditions so you can run the SAME question through
  each and compare:
    - index_condition="mono"  -> monolingual index (only this language)
    - index_condition="multi" -> multilingual shared index (any language)

Usage:
  python src/pipeline.py "What is the eligibility for the scholarship?" en mono
  python src/pipeline.py "What is the eligibility for the scholarship?" en multi
"""

import truststore
truststore.inject_into_ssl()

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CHROMA_MONO_DIR, CHROMA_MULTI_DIR, EMBEDDING_MODEL_NAME, EVAL_DB_PATH, TOP_K
from src.generator import generate_answer
from src.verifier import verify_answer


SCHEMA = """
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    question TEXT NOT NULL,
    language TEXT NOT NULL,
    index_condition TEXT NOT NULL,       -- 'mono' or 'multi'
    verifier_enabled INTEGER NOT NULL DEFAULT 1,  -- 1 = verifier ran, 0 = skipped (for RQ2 ablation)
    retrieved_chunk_ids TEXT NOT NULL,   -- JSON list
    retrieved_chunks TEXT NOT NULL,      -- JSON list (full text, for inspection)
    retrieved_languages TEXT NOT NULL,   -- JSON list (which language each retrieved chunk was in)
    answer TEXT NOT NULL,
    verifier_verdict TEXT NOT NULL,      -- SUPPORTED / UNSUPPORTED / PARTIALLY_SUPPORTED / NOT_VERIFIED
    verifier_explanation TEXT,
    verifier_unsupported_claims TEXT,    -- JSON list
    retrieval_latency_ms REAL,
    generation_latency_ms REAL,
    verification_latency_ms REAL,
    total_latency_ms REAL
);
"""


def init_db():
    EVAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(EVAL_DB_PATH)
    conn.execute(SCHEMA)
    # Migration: if the DB already existed from before this column was added,
    # add it now. Harmless no-op if the column is already there.
    try:
        conn.execute("ALTER TABLE logs ADD COLUMN verifier_enabled INTEGER NOT NULL DEFAULT 1")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()
    conn.close()


def retrieve(question: str, lang: str, index_condition: str, model):
    """Retrieve top-k chunks under the given index condition. Returns (chunks, ids, langs, latency_ms)."""
    import chromadb

    start = time.perf_counter()
    query_embedding = model.encode([f"query: {question}"], convert_to_numpy=True).tolist()

    if index_condition == "mono":
        client = chromadb.PersistentClient(path=str(CHROMA_MONO_DIR))
        collection_name = f"lang_{lang}"
    elif index_condition == "multi":
        client = chromadb.PersistentClient(path=str(CHROMA_MULTI_DIR))
        collection_name = "all_languages"
    else:
        raise ValueError("index_condition must be 'mono' or 'multi'")

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        raise RuntimeError(
            f"Could not find index collection '{collection_name}' "
            f"(index_condition='{index_condition}', language='{lang}'). "
            f"Have you run `python src/embed_index.py` yet? "
            f"If you're testing a new language, make sure it's been ingested and indexed first."
        )

    results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)
    latency_ms = (time.perf_counter() - start) * 1000

    chunk_ids = results["ids"][0]
    chunks = results["documents"][0]
    langs = [m["language"] for m in results["metadatas"][0]]

    return chunks, chunk_ids, langs, latency_ms


def run_pipeline(question: str, lang: str, index_condition: str = "mono", verifier_enabled: bool = True) -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    total_start = time.perf_counter()

    # 1. Retrieve
    chunks, chunk_ids, retrieved_langs, retrieval_ms = retrieve(question, lang, index_condition, model)

    # 2. Generate
    gen_start = time.perf_counter()
    answer = generate_answer(question, chunks, lang)
    generation_ms = (time.perf_counter() - gen_start) * 1000

    # 3. Verify (skippable — this is the RQ2 ablation: does the verifier
    # actually help? To measure that, you need answers WITHOUT it too.)
    if verifier_enabled:
        verify_start = time.perf_counter()
        verdict = verify_answer(question, chunks, answer)
        verification_ms = (time.perf_counter() - verify_start) * 1000
    else:
        verdict = {
            "verdict": "NOT_VERIFIED",
            "explanation": "Verifier was disabled for this run.",
            "unsupported_claims": [],
        }
        verification_ms = 0.0

    total_ms = (time.perf_counter() - total_start) * 1000

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "language": lang,
        "index_condition": index_condition,
        "verifier_enabled": verifier_enabled,
        "retrieved_chunk_ids": chunk_ids,
        "retrieved_chunks": chunks,
        "retrieved_languages": retrieved_langs,
        "answer": answer,
        "verifier_verdict": verdict.get("verdict", "PARSE_ERROR"),
        "verifier_explanation": verdict.get("explanation", ""),
        "verifier_unsupported_claims": verdict.get("unsupported_claims", []),
        "retrieval_latency_ms": retrieval_ms,
        "generation_latency_ms": generation_ms,
        "verification_latency_ms": verification_ms,
        "total_latency_ms": total_ms,
    }

    log_to_db(record)
    return record


def log_to_db(record: dict):
    init_db()
    conn = sqlite3.connect(EVAL_DB_PATH)
    conn.execute(
        """
        INSERT INTO logs (
            timestamp, question, language, index_condition, verifier_enabled,
            retrieved_chunk_ids, retrieved_chunks, retrieved_languages,
            answer, verifier_verdict, verifier_explanation, verifier_unsupported_claims,
            retrieval_latency_ms, generation_latency_ms, verification_latency_ms, total_latency_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["timestamp"],
            record["question"],
            record["language"],
            record["index_condition"],
            int(record["verifier_enabled"]),
            json.dumps(record["retrieved_chunk_ids"], ensure_ascii=False),
            json.dumps(record["retrieved_chunks"], ensure_ascii=False),
            json.dumps(record["retrieved_languages"], ensure_ascii=False),
            record["answer"],
            record["verifier_verdict"],
            record["verifier_explanation"],
            json.dumps(record["verifier_unsupported_claims"], ensure_ascii=False),
            record["retrieval_latency_ms"],
            record["generation_latency_ms"],
            record["verification_latency_ms"],
            record["total_latency_ms"],
        ),
    )
    conn.commit()
    conn.close()


def main():
    if len(sys.argv) < 3:
        print('Usage: python src/pipeline.py "your question" <lang: en|hi|bn> [mono|multi] [verify|noverify]')
        sys.exit(1)

    question = sys.argv[1]
    lang = sys.argv[2]
    index_condition = sys.argv[3] if len(sys.argv) > 3 else "mono"
    verifier_enabled = (sys.argv[4] != "noverify") if len(sys.argv) > 4 else True

    print(f"Question: {question!r}  (language: {lang}, index: {index_condition}, verifier: {'ON' if verifier_enabled else 'OFF'})")
    print("Running pipeline...\n")

    record = run_pipeline(question, lang, index_condition, verifier_enabled)

    print("=== Answer ===")
    print(record["answer"])
    print(f"\n=== Verifier verdict: {record['verifier_verdict']} ===")
    print(record["verifier_explanation"])
    print(f"\n=== Latency ===")
    print(f"  Retrieval:    {record['retrieval_latency_ms']:.1f} ms")
    print(f"  Generation:   {record['generation_latency_ms']:.1f} ms")
    print(f"  Verification: {record['verification_latency_ms']:.1f} ms")
    print(f"  Total:        {record['total_latency_ms']:.1f} ms")
    print(f"\nLogged to {EVAL_DB_PATH}")


if __name__ == "__main__":
    main()