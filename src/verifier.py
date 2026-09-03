"""
Step 5: Verifier agent.

What this does:
  Takes the generator's answer + the same context chunks it was given, and
  makes a SEPARATE LLM call (Groq, per the project's multi-provider design)
  whose only job is to judge: is this answer actually supported by the
  context, or does it contain claims not backed by it (i.e. hallucination)?

  This is your RQ2 in code: "Does a verifier agent reduce hallucination
  rate, and by how much, per language?" To answer that later, you need to
  log the verifier's verdict for every generated answer — that logging
  happens in the eval harness (a later step), this script just produces
  the verdict.

  Verdict format: the verifier returns one of:
    - "SUPPORTED"     — answer is fully backed by the context
    - "UNSUPPORTED"   — answer contains claims not found in the context
    - "PARTIALLY_SUPPORTED" — some claims backed, some not

  along with a short explanation (useful for your report/appendix, and for
  debugging when something looks wrong).

Usage (standalone test):
  python src/verifier.py "What is the eligibility for the scholarship?" en
  (this runs retrieval -> generation -> verification, all in one, for a
  quick end-to-end check)
"""

import truststore
truststore.inject_into_ssl()  # must run before any network calls

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import GROQ_MODEL_NAME

load_dotenv()

VALID_VERDICTS = {"SUPPORTED", "UNSUPPORTED", "PARTIALLY_SUPPORTED"}


def build_verifier_prompt(question: str, context_chunks: list[str], answer: str) -> str:
    context_block = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(context_chunks))

    prompt = f"""You are a strict fact-checker. Your job is to check whether an AI-generated answer is fully supported by the given context, or whether it contains claims not found in the context (hallucination).

Context:
{context_block}

Question: {question}

Generated Answer:
{answer}

Instructions:
- Check EVERY factual claim in the Generated Answer against the Context.
- Respond with ONLY a JSON object in this exact format, nothing else:
{{"verdict": "SUPPORTED" | "UNSUPPORTED" | "PARTIALLY_SUPPORTED", "explanation": "brief reason, 1-2 sentences", "unsupported_claims": ["list any specific claims not found in context, empty list if none"]}}

Respond with ONLY the JSON object, no markdown formatting, no extra text."""
    return prompt


def verify_answer(question: str, context_chunks: list[str], answer: str) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)

    prompt = build_verifier_prompt(question, context_chunks, answer)
    completion = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic — we want consistent judging, not creativity
    )

    raw = completion.choices[0].message.content.strip()

    # Strip markdown code fences if the model added them despite instructions
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: if the model didn't return clean JSON, don't crash the
        # whole pipeline — log it as unparseable so you can inspect later.
        result = {
            "verdict": "PARSE_ERROR",
            "explanation": f"Could not parse verifier output as JSON. Raw output: {raw[:300]}",
            "unsupported_claims": [],
        }

    if result.get("verdict") not in VALID_VERDICTS:
        result.setdefault("verdict", "PARSE_ERROR")

    return result


def main():
    if len(sys.argv) < 3:
        print('Usage: python src/verifier.py "your question" <lang: en|hi|bn>')
        sys.exit(1)

    question = sys.argv[1]
    lang = sys.argv[2]

    # Reuse the generator's retrieve+generate step so this is a full
    # end-to-end test: retrieval -> generation -> verification
    from src.generator import retrieve_and_generate

    print(f"Question: {question!r}  (language: {lang})")
    print("Retrieving context + generating answer...\n")
    gen_result = retrieve_and_generate(question, lang)

    print("=== Generated answer ===")
    print(gen_result["answer"])

    print("\nVerifying answer against context...\n")
    verdict = verify_answer(question, gen_result["retrieved_chunks"], gen_result["answer"])

    print("=== Verifier verdict ===")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()