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
from src.generator import generate_answer, regenerate_answer
from src.verifier import verify_answer

# Verdicts that trigger a regeneration attempt when regenerate_on_failure=True.
# NOT_VERIFIED/PARSE_ERROR are excluded — regenerating without a real verdict
# to react to would just be a second random guess, not a targeted correction.
REGENERATION_TRIGGER_VERDICTS = {"UNSUPPORTED", "PARTIALLY_SUPPORTED"}


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
    regeneration_latency_ms REAL DEFAULT 0,
    total_latency_ms REAL,
    was_regenerated INTEGER NOT NULL DEFAULT 0,  -- 1 = answer was corrected after a failed verification
    original_answer TEXT,                         -- the pre-regeneration answer, if regenerated
    original_verdict TEXT                          -- the verdict that triggered regeneration, if any
);
"""


def init_db():
    EVAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(EVAL_DB_PATH)
    conn.execute(SCHEMA)
    # Migration: if the DB already existed from before these columns were
    # added, add them now. Harmless no-op if a column already exists.
    for statement in [
        "ALTER TABLE logs ADD COLUMN verifier_enabled INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE logs ADD COLUMN regeneration_latency_ms REAL DEFAULT 0",
        "ALTER TABLE logs ADD COLUMN was_regenerated INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE logs ADD COLUMN original_answer TEXT",
        "ALTER TABLE logs ADD COLUMN original_verdict TEXT",
    ]:
        try:
            conn.execute(statement)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()


def retrieve(question: str, lang: str, index_condition: str, model):
    """Retrieve top-k chunks under the given index condition. Returns (chunks, ids, langs, scheme_slugs, latency_ms)."""
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
    scheme_slugs = [m.get("scheme_slug", "") for m in results["metadatas"][0]]

    return chunks, chunk_ids, langs, scheme_slugs, latency_ms


def run_pipeline(
    question: str, lang: str, index_condition: str = "mono",
    verifier_enabled: bool = True, regenerate_on_failure: bool = True,
) -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    total_start = time.perf_counter()

    # 1. Retrieve
    chunks, chunk_ids, retrieved_langs, retrieved_scheme_slugs, retrieval_ms = retrieve(question, lang, index_condition, model)

    # 2. Generate
    gen_start = time.perf_counter()
    answer = generate_answer(question, chunks, lang)
    generation_ms = (time.perf_counter() - gen_start) * 1000

    # 3. Verify (skippable — this is the RQ2 ablation: does the verifier
    # actually help? To measure that, you need answers WITHOUT it too.)
    was_regenerated = False
    original_answer = None
    original_verdict = None
    regeneration_ms = 0.0

    if verifier_enabled:
        verify_start = time.perf_counter()
        verdict = verify_answer(question, chunks, answer)
        verification_ms = (time.perf_counter() - verify_start) * 1000

        # 3b. Regenerate on failure — this is the fix for the "verifier only
        # labels, doesn't improve" gap. If the verifier flagged the answer,
        # give the generator specific feedback (what was unsupported, and
        # why) and ask for a corrected answer, then re-verify the result.
        # We only do this ONCE per question — no infinite correction loops.
        if regenerate_on_failure and verdict.get("verdict") in REGENERATION_TRIGGER_VERDICTS:
            regen_start = time.perf_counter()
            original_answer = answer
            original_verdict = verdict.get("verdict")

            answer = regenerate_answer(
                question, chunks, lang,
                previous_answer=original_answer,
                unsupported_claims=verdict.get("unsupported_claims", []),
                verifier_explanation=verdict.get("explanation", ""),
            )
            verdict = verify_answer(question, chunks, answer)  # re-check the corrected answer
            was_regenerated = True
            regeneration_ms = (time.perf_counter() - regen_start) * 1000
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
        "retrieved_scheme_slugs": retrieved_scheme_slugs,
        "answer": answer,
        "verifier_verdict": verdict.get("verdict", "PARSE_ERROR"),
        "verifier_explanation": verdict.get("explanation", ""),
        "verifier_unsupported_claims": verdict.get("unsupported_claims", []),
        "retrieval_latency_ms": retrieval_ms,
        "generation_latency_ms": generation_ms,
        "verification_latency_ms": verification_ms,
        "regeneration_latency_ms": regeneration_ms,
        "total_latency_ms": total_ms,
        "was_regenerated": was_regenerated,
        "original_answer": original_answer,
        "original_verdict": original_verdict,
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
            retrieval_latency_ms, generation_latency_ms, verification_latency_ms,
            regeneration_latency_ms, total_latency_ms,
            was_regenerated, original_answer, original_verdict
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            record["regeneration_latency_ms"],
            record["total_latency_ms"],
            int(record["was_regenerated"]),
            record["original_answer"],
            record["original_verdict"],
        ),
    )
    conn.commit()
    conn.close()


def main():
    if len(sys.argv) < 3:
        print('Usage: python src/pipeline.py "your question" <lang: en|hi|bn> [mono|multi] [verify|noverify] [regen|noregen]')
        sys.exit(1)

    question = sys.argv[1]
    lang = sys.argv[2]
    index_condition = sys.argv[3] if len(sys.argv) > 3 else "mono"
    verifier_enabled = (sys.argv[4] != "noverify") if len(sys.argv) > 4 else True
    regenerate_on_failure = (sys.argv[5] != "noregen") if len(sys.argv) > 5 else True

    print(f"Question: {question!r}  (language: {lang}, index: {index_condition}, "
          f"verifier: {'ON' if verifier_enabled else 'OFF'}, regenerate: {'ON' if regenerate_on_failure else 'OFF'})")
    print("Running pipeline...\n")

    record = run_pipeline(question, lang, index_condition, verifier_enabled, regenerate_on_failure)

    if record["was_regenerated"]:
        print("=== Original answer (flagged by verifier) ===")
        print(record["original_answer"])
        print(f"\n=== Original verdict: {record['original_verdict']} ===")
        print("\n--- Regenerated after corrective feedback ---\n")

    print("=== Final answer ===")
    print(record["answer"])
    print(f"\n=== Verifier verdict: {record['verifier_verdict']} ===")
    print(record["verifier_explanation"])
    print(f"\n=== Latency ===")
    print(f"  Retrieval:    {record['retrieval_latency_ms']:.1f} ms")
    print(f"  Generation:   {record['generation_latency_ms']:.1f} ms")
    print(f"  Verification: {record['verification_latency_ms']:.1f} ms")
    if record["was_regenerated"]:
        print(f"  Regeneration: {record['regeneration_latency_ms']:.1f} ms")
    print(f"  Total:        {record['total_latency_ms']:.1f} ms")
    print(f"\nLogged to {EVAL_DB_PATH}")


if __name__ == "__main__":
    main()