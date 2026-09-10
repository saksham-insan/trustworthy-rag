"""
Step 8: Evaluation harness.

Runs every question in eval/test_set.json through the pipeline across
THREE conditions, so you get real numbers for RQ1-RQ4:

  A) mono,  verifier ON   (45 runs) — baseline: per-language retrieval + hallucination check
  B) multi, verifier ON   (45 runs) — RQ3: does mixing languages in one index hurt retrieval?
  C) mono,  verifier OFF  (45 runs) — RQ2: what changes when the verifier is skipped?

Total: ~135 pipeline runs. This WILL take a while (each run involves 2-3 API
calls with possible retries) — expect 15-30+ minutes depending on API speed
and rate limits. That's normal, not a bug.

Robustness: results are written incrementally (one line per completed run)
to a timestamped .jsonl file, so if this crashes or you Ctrl+C partway
through, everything completed so far is already saved to disk — nothing
is lost. Every run is also still logged to eval/logs.sqlite as usual via
pipeline.run_pipeline().

Automatic grading (no manual answer-reading required):
  - retrieval_correct: True if any of the top-k retrieved chunks came from
    the question's expected_scheme_slug (ground truth from test_set.json)
  - keyword_coverage: fraction of expected_keywords found (case-insensitive,
    comma-insensitive) in the generated answer text

Usage:
  python src/run_eval.py
  python src/run_eval.py --limit 3     # quick test: only first 3 question groups
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROJECT_ROOT
from src.pipeline import run_pipeline

TEST_SET_PATH = PROJECT_ROOT / "eval" / "test_set.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

# The three conditions this harness tests. Each is (index_condition, verifier_enabled, run_label).
RUN_CONDITIONS = [
    ("mono", True, "mono_verified"),
    ("multi", True, "multi_verified"),
    ("mono", False, "mono_unverified"),
]

LANGUAGES = ["en", "hi", "bn"]


def normalize(text: str) -> str:
    """Lowercase and strip commas/spaces so '3,50,000' matches '3 50 000' etc."""
    return re.sub(r"[,\s]", "", text.lower())


def compute_keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return None  # no ground truth to check against
    norm_answer = normalize(answer)
    hits = sum(1 for kw in expected_keywords if normalize(kw) in norm_answer)
    return hits / len(expected_keywords)


def load_test_set() -> list[dict]:
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(f"{TEST_SET_PATH} not found.")
    return json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))


def run_one(entry: dict, lang: str, index_condition: str, verifier_enabled: bool, run_label: str) -> dict:
    question = entry["questions"][lang]
    expected_scheme = entry["scheme_slug"]
    expected_keywords = entry["expected_keywords"]

    try:
        record = run_pipeline(question, lang, index_condition, verifier_enabled)
        retrieved_schemes = record.get("retrieved_scheme_slugs", [])
        retrieval_correct = expected_scheme in retrieved_schemes
        keyword_coverage = compute_keyword_coverage(record["answer"], expected_keywords)

        was_regenerated = record.get("was_regenerated", False)
        original_keyword_coverage = None
        if was_regenerated and record.get("original_answer"):
            # Lets us directly compare: did regeneration actually improve
            # factual coverage, or just change the wording?
            original_keyword_coverage = compute_keyword_coverage(record["original_answer"], expected_keywords)

        return {
            "run_label": run_label,
            "question_id": entry["id"],
            "scheme_slug": expected_scheme,
            "category": entry["category"],
            "language": lang,
            "index_condition": index_condition,
            "verifier_enabled": verifier_enabled,
            "question": question,
            "answer": record["answer"],
            "retrieved_scheme_slugs": retrieved_schemes,
            "retrieval_correct": retrieval_correct,
            "keyword_coverage": keyword_coverage,
            "verifier_verdict": record["verifier_verdict"],
            "was_regenerated": was_regenerated,
            "original_verdict": record.get("original_verdict"),
            "original_keyword_coverage": original_keyword_coverage,
            "retrieval_latency_ms": record["retrieval_latency_ms"],
            "generation_latency_ms": record["generation_latency_ms"],
            "verification_latency_ms": record["verification_latency_ms"],
            "regeneration_latency_ms": record.get("regeneration_latency_ms", 0.0),
            "total_latency_ms": record["total_latency_ms"],
            "error": None,
        }
    except Exception as e:
        # Don't let one failed question kill the whole 135-run batch — log
        # the failure and keep going. You can review failures afterward.
        return {
            "run_label": run_label,
            "question_id": entry["id"],
            "scheme_slug": expected_scheme,
            "category": entry["category"],
            "language": lang,
            "index_condition": index_condition,
            "verifier_enabled": verifier_enabled,
            "question": question,
            "answer": None,
            "retrieved_scheme_slugs": None,
            "retrieval_correct": None,
            "keyword_coverage": None,
            "verifier_verdict": None,
            "was_regenerated": None,
            "original_verdict": None,
            "original_keyword_coverage": None,
            "retrieval_latency_ms": None,
            "generation_latency_ms": None,
            "verification_latency_ms": None,
            "regeneration_latency_ms": None,
            "total_latency_ms": None,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run the evaluation harness over the test set.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N question groups (for a quick test run).")
    args = parser.parse_args()

    test_set = load_test_set()
    if args.limit:
        test_set = test_set[: args.limit]

    total_runs = len(test_set) * len(LANGUAGES) * len(RUN_CONDITIONS)
    print(f"Loaded {len(test_set)} question groups x {len(LANGUAGES)} languages x "
          f"{len(RUN_CONDITIONS)} conditions = {total_runs} total runs.")
    print("This will take a while — results are saved incrementally, so it's safe to Ctrl+C and resume later.\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"eval_run_{timestamp}.jsonl"

    completed = 0
    failed = 0
    start_time = time.perf_counter()

    with open(out_path, "a", encoding="utf-8") as f:
        for index_condition, verifier_enabled, run_label in RUN_CONDITIONS:
            for entry in test_set:
                for lang in LANGUAGES:
                    completed += 1
                    label = f"[{completed}/{total_runs}] {run_label} | {entry['id']} | {lang}"
                    print(label, end="  ", flush=True)

                    result = run_one(entry, lang, index_condition, verifier_enabled, run_label)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()  # write immediately — don't lose progress on crash

                    if result["error"]:
                        failed += 1
                        print(f"FAILED: {result['error'][:100]}")
                    else:
                        rc = "CORRECT" if result["retrieval_correct"] else "WRONG"
                        kc = result["keyword_coverage"]
                        kc_str = f"{kc:.0%}" if kc is not None else "n/a"
                        regen_str = " [REGENERATED]" if result["was_regenerated"] else ""
                        print(f"retrieval={rc} keywords={kc_str} verdict={result['verifier_verdict']}{regen_str}")

    elapsed_min = (time.perf_counter() - start_time) / 60
    print(f"\nDone. {completed - failed}/{completed} runs succeeded ({failed} failed) in {elapsed_min:.1f} minutes.")
    print(f"Results saved to: {out_path}")
    print(f"\nNext: python src/analyze_eval.py {out_path.name}")


if __name__ == "__main__":
    main()