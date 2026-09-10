"""
Step 9: Analyze evaluation results.

Reads a .jsonl file produced by run_eval.py and computes the actual
numbers for your research questions:

  RQ1 — retrieval accuracy by language (within mono_verified condition)
  RQ2 — verifier verdict distribution (verified) vs keyword coverage
        with verifier ON vs OFF (mono_verified vs mono_unverified)
  RQ3 — retrieval accuracy: mono_verified vs multi_verified, per language
  RQ4 — latency by condition (retrieval / generation / verification / total)

Usage:
  python src/analyze_eval.py                          # uses the most recent eval_run_*.jsonl
  python src/analyze_eval.py eval_run_20260907T....jsonl
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def load_results(filename: str = None) -> list[dict]:
    if filename:
        path = RESULTS_DIR / filename
    else:
        candidates = sorted(RESULTS_DIR.glob("eval_run_*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No eval_run_*.jsonl files found in {RESULTS_DIR}. Run src/run_eval.py first.")
        path = candidates[-1]  # most recent

    print(f"Analyzing: {path.name}\n")
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results


def pct(n, d):
    return f"{(n / d * 100):.0f}%" if d else "n/a"


def avg(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def rq1_retrieval_by_language(results):
    print("=" * 70)
    print("RQ1 - Retrieval accuracy by language (condition: mono, verifier ON)")
    print("=" * 70)
    subset = [r for r in results if r["run_label"] == "mono_verified" and r["error"] is None]
    by_lang = defaultdict(list)
    for r in subset:
        by_lang[r["language"]].append(r["retrieval_correct"])
    for lang in ["en", "hi", "bn"]:
        vals = by_lang.get(lang, [])
        correct = sum(1 for v in vals if v)
        print(f"  {lang}: {correct}/{len(vals)} correct  ({pct(correct, len(vals))})")
    print()


def rq3_mono_vs_multi(results):
    print("=" * 70)
    print("RQ3 - Monolingual vs multilingual index, retrieval accuracy by language")
    print("=" * 70)
    for lang in ["en", "hi", "bn"]:
        mono = [r for r in results if r["run_label"] == "mono_verified" and r["language"] == lang and r["error"] is None]
        multi = [r for r in results if r["run_label"] == "multi_verified" and r["language"] == lang and r["error"] is None]
        mono_correct = sum(1 for r in mono if r["retrieval_correct"])
        multi_correct = sum(1 for r in multi if r["retrieval_correct"])
        print(f"  {lang}: mono={pct(mono_correct, len(mono))} ({mono_correct}/{len(mono)})   "
              f"multi={pct(multi_correct, len(multi))} ({multi_correct}/{len(multi)})")
    print()


def rq2_verifier_effect(results):
    print("=" * 70)
    print("RQ2 - Verifier effect (mono condition: verifier ON vs OFF)")
    print("=" * 70)
    verified = [r for r in results if r["run_label"] == "mono_verified" and r["error"] is None]
    unverified = [r for r in results if r["run_label"] == "mono_unverified" and r["error"] is None]

    print("  Verifier verdict distribution (verifier ON):")
    verdict_counts = defaultdict(int)
    for r in verified:
        verdict_counts[r["verifier_verdict"]] += 1
    for verdict, count in sorted(verdict_counts.items()):
        print(f"    {verdict}: {count}/{len(verified)} ({pct(count, len(verified))})")

    verified_kw = avg([r["keyword_coverage"] for r in verified if r["keyword_coverage"] is not None])
    unverified_kw = avg([r["keyword_coverage"] for r in unverified if r["keyword_coverage"] is not None])
    v_str = f"{verified_kw:.0%}" if verified_kw is not None else "n/a"
    u_str = f"{unverified_kw:.0%}" if unverified_kw is not None else "n/a"
    print(f"\n  Avg keyword coverage - verifier ON:  {v_str}")
    print(f"  Avg keyword coverage - verifier OFF: {u_str}")
    print()


def rq4_latency(results):
    print("=" * 70)
    print("RQ4 - Latency by condition (avg ms)")
    print("=" * 70)
    for label in ["mono_verified", "multi_verified", "mono_unverified"]:
        subset = [r for r in results if r["run_label"] == label and r["error"] is None]
        if not subset:
            continue
        retr = avg([r["retrieval_latency_ms"] for r in subset])
        gen = avg([r["generation_latency_ms"] for r in subset])
        ver = avg([r["verification_latency_ms"] for r in subset])
        tot = avg([r["total_latency_ms"] for r in subset])
        print(f"  {label:<18} retrieval={retr:6.0f}ms  generation={gen:6.0f}ms  "
              f"verification={ver:6.0f}ms  total={tot:6.0f}ms  (n={len(subset)})")
    print()


def failures_summary(results):
    failed = [r for r in results if r["error"]]
    if failed:
        print("=" * 70)
        print(f"FAILURES ({len(failed)} runs failed - review these separately)")
        print("=" * 70)
        for r in failed[:10]:
            print(f"  {r['run_label']} | {r['question_id']} | {r['language']}: {r['error'][:120]}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
        print()


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else None
    results = load_results(filename)

    total = len(results)
    errored = sum(1 for r in results if r["error"])
    print(f"Total runs: {total}  ({total - errored} succeeded, {errored} failed)\n")

    rq1_retrieval_by_language(results)
    rq3_mono_vs_multi(results)
    rq2_verifier_effect(results)
    rq4_latency(results)
    failures_summary(results)


if __name__ == "__main__":
    main()