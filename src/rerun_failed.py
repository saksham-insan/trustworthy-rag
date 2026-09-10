"""
Re-run only the FAILED entries from a previous evaluation run, and patch
them into the results file in place — so you end up with one clean file
covering all 135 runs, without re-doing the ~128 that already succeeded.

Typical use: some runs failed due to a free-tier daily quota limit (Groq,
Gemini). Wait for the quota to reset (usually ~24h, check the provider's
error message for the exact retry time), then run this.

Usage:
  python src/rerun_failed.py                          # uses the most recent eval_run_*.jsonl
  python src/rerun_failed.py eval_run_20260907....jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROJECT_ROOT
from src.run_eval import load_test_set, run_one

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def find_latest_results_file(filename: str = None) -> Path:
    if filename:
        return RESULTS_DIR / filename
    candidates = sorted(RESULTS_DIR.glob("eval_run_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No eval_run_*.jsonl files found in {RESULTS_DIR}.")
    return candidates[-1]


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else None
    results_path = find_latest_results_file(filename)

    print(f"Loading: {results_path.name}")
    rows = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    failed_indices = [i for i, r in enumerate(rows) if r["error"]]
    if not failed_indices:
        print("No failed entries found — nothing to re-run. This file is already clean.")
        return

    print(f"Found {len(failed_indices)} failed entries. Re-running them now...\n")

    test_set = load_test_set()
    test_set_by_id = {entry["id"]: entry for entry in test_set}

    still_failed = 0
    fixed = 0

    for i in failed_indices:
        old_row = rows[i]
        entry = test_set_by_id.get(old_row["question_id"])
        if entry is None:
            print(f"  [skip] question_id '{old_row['question_id']}' not found in current test_set.json — skipping.")
            continue

        label = f"{old_row['run_label']} | {old_row['question_id']} | {old_row['language']}"
        print(f"Retrying: {label} ...", end="  ", flush=True)

        new_result = run_one(
            entry,
            old_row["language"],
            old_row["index_condition"],
            old_row["verifier_enabled"],
            old_row["run_label"],
        )
        rows[i] = new_result  # patch in place, preserving original position/order

        if new_result["error"]:
            still_failed += 1
            print(f"still failing: {new_result['error'][:100]}")
        else:
            fixed += 1
            rc = "CORRECT" if new_result["retrieval_correct"] else "WRONG"
            print(f"OK — retrieval={rc} verdict={new_result['verifier_verdict']}")

    # Write back to the SAME file — now with (hopefully) fewer failures
    with open(results_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_errors_now = sum(1 for r in rows if r["error"])
    print(f"\nDone. Fixed {fixed}, still failing {still_failed}.")
    print(f"Total runs in file: {len(rows)} ({len(rows) - total_errors_now} succeeded, {total_errors_now} failed)")
    print(f"Patched file: {results_path}")
    print(f"\nNext: python src/analyze_eval.py {results_path.name}")


if __name__ == "__main__":
    main()