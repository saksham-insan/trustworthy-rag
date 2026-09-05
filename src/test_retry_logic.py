"""
Test the retry-with-backoff logic in isolation, WITHOUT needing to
actually trigger a real rate limit (which is slow and unreliable to test
against). Uses fake functions that raise controlled errors to prove:

  1. Transient errors (rate limits, timeouts) get retried and eventually
     succeed if the underlying call recovers.
  2. Non-transient errors (bad API key, invalid request) fail IMMEDIATELY
     without wasting time retrying something that can never succeed.
  3. If a transient error never recovers, we still eventually give up
     (not stuck retrying forever).

Usage:
  python src/test_retry_logic.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import retry_with_backoff


def test_recovers_after_transient_failures():
    """Fails twice with a rate-limit-style error, then succeeds. Should retry and pass."""
    attempts = {"count": 0}

    def flaky_call():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise Exception("429 rate limit exceeded, please retry")
        return "success"

    result = retry_with_backoff(flaky_call, label="test-recovers", base_delay_seconds=0.1)
    assert result == "success", f"Expected 'success', got {result!r}"
    assert attempts["count"] == 3, f"Expected 3 attempts, got {attempts['count']}"
    print("[PASS] Recovers after transient failures (took 3 attempts, as expected)")


def test_fails_fast_on_permanent_error():
    """Fails with an auth error — should NOT retry at all, should raise immediately."""
    attempts = {"count": 0}

    def broken_call():
        attempts["count"] += 1
        raise Exception("401 Unauthorized: invalid API key")

    start = time.perf_counter()
    try:
        retry_with_backoff(broken_call, label="test-fails-fast", base_delay_seconds=5.0)
        raise AssertionError("Expected an exception to be raised, but none was")
    except Exception as e:
        if "Unauthorized" not in str(e):
            raise
    elapsed = time.perf_counter() - start
    assert attempts["count"] == 1, f"Expected exactly 1 attempt (no retries), got {attempts['count']}"
    assert elapsed < 1.0, f"Expected near-instant failure, took {elapsed:.1f}s (means it retried when it shouldn't have)"
    print(f"[PASS] Fails fast on permanent error (1 attempt, {elapsed:.2f}s — no wasted retries)")


def test_gives_up_after_max_retries():
    """Always fails with a transient error — should retry up to the limit, then raise."""
    attempts = {"count": 0}

    def always_flaky():
        attempts["count"] += 1
        raise Exception("503 Service Unavailable")

    try:
        retry_with_backoff(always_flaky, label="test-gives-up", max_retries=3, base_delay_seconds=0.1)
        raise AssertionError("Expected an exception after exhausting retries, but none was raised")
    except Exception as e:
        if "Service Unavailable" not in str(e):
            raise
    assert attempts["count"] == 3, f"Expected exactly 3 attempts (max_retries), got {attempts['count']}"
    print("[PASS] Gives up after max_retries (3 attempts, then raised — didn't retry forever)")


if __name__ == "__main__":
    print("Testing retry_with_backoff logic...\n")
    test_recovers_after_transient_failures()
    test_fails_fast_on_permanent_error()
    test_gives_up_after_max_retries()
    print("\nAll retry logic tests passed.")