# Phase 1 Test Set

`test_set.json` — 15 question groups × 3 languages (English, Hindi, Bengali) = 45 question
instances, covering all 5 schemes in your current corpus, 3 questions per scheme
(eligibility / benefits / application process).

## Structure

Each entry:
```json
{
  "id": "pmusp_eligibility",
  "scheme_slug": "pmusp",              // ground truth — used to auto-check retrieval accuracy
  "category": "eligibility",
  "questions": { "en": "...", "hi": "...", "bn": "..." },
  "expected_keywords": ["80th percentile", "4,50,000", "AICTE"],  // for auto-checking answer quality
  "notes": "..."
}
```

## Why `scheme_slug` matters (this is the automatic grading trick)

When the evaluation harness runs a question through the pipeline, we already know which
`scheme_slug` the retrieved chunks came from (it's in the chunk metadata). If the question
is `pmusp_eligibility` and the pipeline retrieves chunks tagged `scheme_slug: "pmusp"` — the
retrieval worked correctly. If it retrieves chunks from a *different* scheme, retrieval
failed for that question. **This gives you real RQ1 (accuracy drop by language) and RQ3
(monolingual vs multilingual interference) numbers without manually judging each answer.**

## Why `expected_keywords` matters

A lightweight, automatic proxy for "is the generated answer actually correct" — after
running a question, we can check what fraction of `expected_keywords` show up in the
generated answer. It's not perfect (a good answer could phrase things differently), but for
factual government-scheme content (specific numbers, named portals, named categories) it's
a reasonable, zero-manual-effort signal, especially combined with the verifier's own verdict.

## What's left for you to fill in

7 of the 15 entries have **empty `expected_keywords`** — for PMVS, PM-USPY, and PM-YASASVI,
I only saw fragments of your source content during earlier testing, not the full text, so I
deliberately did NOT invent numbers I wasn't sure about (that would corrupt your evaluation
ground truth).

**To fill these in (~10 minutes total):** open your own files —
```
data/raw/en/pmvs.txt
data/raw/en/pm-uspy.txt
data/raw/en/pm-yasasvi.txt
```
— and for each, pick 2-3 short, distinctive facts (an exact ₹ amount, a percentage, a named
portal/institution) that a correct answer to that question should mention. Add them to the
matching entry's `expected_keywords` list in `test_set.json`. Keep the `notes` field's TODO
text or replace it with a short note on where you got the fact from — doesn't need to be
formal, just enough that you remember later.

## What this test set does NOT do yet

- No **Benglish** (Bengali-English code-mixed) questions yet — add these once you're ready
  to test that condition; the JSON structure would need a 4th language key (`"benglish"`) on
  each entry.
- No manual "correct answer" text — deliberately skipped for phase-1 scale (45 hand-written
  gold answers isn't worth it yet). The evaluation harness (next step, once you're ready)
  will use `scheme_slug` + `expected_keywords` + the verifier's own verdict as this stage's
  measurement approach — a fair method for a Phase 1 review, and easy to upgrade later.
- Only 5 schemes — once you add more documents, extend this file the same way (one JSON
  object per scheme × category).