"""
Shared utility: retry-with-backoff for API calls.

Why this exists: free-tier LLM APIs (Gemini, Groq) occasionally fail with
TRANSIENT errors that are worth retrying — rate limits (429), momentary
network hiccups, temporary server errors (5xx). Retrying these automatically
means one flaky request doesn't crash a whole batch evaluation run later.

It's equally important NOT to retry errors that will never succeed no
matter how many times you try — an invalid API key, a bad model name, a
malformed request. Retrying those just wastes time and hides the real
problem. This module tries to tell the difference.

Usage:
    from src.utils import retry_with_backoff

    result = retry_with_backoff(
        lambda: client.chat.completions.create(...),
        label="Groq verifier call",
    )
"""

import time
from typing import Callable, TypeVar

T = TypeVar("T")

# Substrings that indicate a RETRYABLE (transient) error, checked
# case-insensitively against the exception message. Not exhaustive, but
# covers what we've actually hit with Gemini/Groq free tiers.
RETRYABLE_SIGNALS = [
    "rate limit",
    "resource_exhausted",
    "429",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "503",
    "502",
    "500",
    "internal server error",
    "overloaded",
]


def is_retryable(error: Exception) -> bool:
    message = str(error).lower()
    return any(signal in message for signal in RETRYABLE_SIGNALS)


def retry_with_backoff(
    fn: Callable[[], T],
    label: str = "API call",
    max_retries: int = 4,
    base_delay_seconds: float = 2.0,
) -> T:
    """
    Calls fn() and retries on transient failures with exponential backoff
    (2s, 4s, 8s, 16s by default). Raises immediately on non-retryable
    errors (no point waiting on those). Raises the last error if all
    retries are exhausted.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if not is_retryable(e):
                print(f"[{label}] Non-retryable error, giving up immediately: {e}")
                raise
            if attempt == max_retries:
                print(f"[{label}] Failed after {max_retries} attempts: {e}")
                raise
            delay = base_delay_seconds * (2 ** (attempt - 1))
            print(f"[{label}] Attempt {attempt}/{max_retries} failed ({e}). Retrying in {delay:.0f}s...")
            time.sleep(delay)

    # Unreachable in practice (loop always returns or raises), but keeps type checkers happy
    raise last_error