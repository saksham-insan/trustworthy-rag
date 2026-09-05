"""
Quick utility: view the logged pipeline runs from eval/logs.sqlite in a
readable table, instead of needing a separate SQLite browser tool.

Usage:
  python src/view_logs.py            # show all logged runs
  python src/view_logs.py 10         # show only the last 10 runs
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import EVAL_DB_PATH


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    if not EVAL_DB_PATH.exists():
        print(f"No log file found at {EVAL_DB_PATH}. Run the pipeline at least once first.")
        return

    conn = sqlite3.connect(EVAL_DB_PATH)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT id, timestamp, question, language, index_condition, verifier_enabled,
               verifier_verdict, retrieval_latency_ms, generation_latency_ms,
               verification_latency_ms, total_latency_ms
        FROM logs
        ORDER BY id DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    rows = conn.execute(query).fetchall()
    conn.close()

    if not rows:
        print("No runs logged yet.")
        return

    print(f"{'ID':<4} {'Lang':<5} {'Cond':<6} {'Verify':<7} {'Verdict':<12} {'Retr(ms)':<10} {'Gen(ms)':<10} {'Verify(ms)':<11} {'Total(ms)':<10} Question")
    print("-" * 140)
    for r in rows:
        q_preview = r["question"][:45] + ("..." if len(r["question"]) > 45 else "")
        verify_flag = "ON" if r["verifier_enabled"] else "OFF"
        print(f"{r['id']:<4} {r['language']:<5} {r['index_condition']:<6} {verify_flag:<7} {r['verifier_verdict']:<12} "
              f"{r['retrieval_latency_ms']:<10.1f} {r['generation_latency_ms']:<10.1f} "
              f"{r['verification_latency_ms']:<11.1f} {r['total_latency_ms']:<10.1f} {q_preview}")

    print(f"\nTotal runs logged: {len(rows)}" + (f" (showing last {limit})" if limit else ""))


if __name__ == "__main__":
    main()