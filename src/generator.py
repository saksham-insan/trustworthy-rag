"""
Step 4: LLM answer generation.

What this does:
  Takes a user question + a list of retrieved context chunks (from
  test_retrieval.py's logic), builds a prompt that instructs the LLM to
  answer ONLY using the given context (not its own general knowledge), and
  calls Gemini to generate the answer in the same language as the question.

  This is the "generator" half of the RAG pipeline — the verifier agent
  (Step 5, next) will later double-check this generator's output against
  the same context before showing anything to the user.

Usage (standalone test):
  python src/generator.py "What is the eligibility for the scholarship?" en
"""

import truststore
truststore.inject_into_ssl()  # must run before any network calls

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    CHROMA_MONO_DIR,
    EMBEDDING_MODEL_NAME,
    GEMINI_MODEL_NAME,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    TOP_K,
)

load_dotenv()

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "bn": "Bengali"}


def build_prompt(question: str, context_chunks: list[str], lang: str) -> str:
    """
    Builds a grounded-answer prompt. Explicitly instructs the model to say
    it doesn't know rather than guess — this matters a lot for your
    hallucination-rate measurements later (RQ2).
    """
    lang_name = LANGUAGE_NAMES.get(lang, "English")
    context_block = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(context_chunks))

    prompt = f"""You are a helpful assistant answering questions about Indian government schemes, using ONLY the context provided below. Do not use any outside knowledge.

Rules:
- Answer strictly using facts stated in the context below.
- If the context does not contain enough information to answer, say clearly that you don't have enough information — do NOT guess or make up an answer.
- Answer in {lang_name}, matching the language of the question.
- Be concise and factual.

Context:
{context_block}

Question: {question}

Answer:"""
    return prompt


def generate_answer(question: str, context_chunks: list[str], lang: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = build_prompt(question, context_chunks, lang)
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
        config={
            "temperature": GENERATION_TEMPERATURE,
            "max_output_tokens": GENERATION_MAX_TOKENS,
        },
    )
    return response.text.strip()


def retrieve_and_generate(question: str, lang: str) -> dict:
    """Convenience wrapper: retrieve top-k chunks (monolingual condition) then generate an answer."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    query_embedding = model.encode([f"query: {question}"], convert_to_numpy=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_MONO_DIR))
    collection = client.get_collection(name=f"lang_{lang}")
    results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)

    context_chunks = results["documents"][0]
    answer = generate_answer(question, context_chunks, lang)

    return {
        "question": question,
        "language": lang,
        "retrieved_chunks": context_chunks,
        "answer": answer,
    }


def main():
    if len(sys.argv) < 3:
        print('Usage: python src/generator.py "your question" <lang: en|hi|bn>')
        sys.exit(1)

    question = sys.argv[1]
    lang = sys.argv[2]

    print(f"Question: {question!r}  (language: {lang})")
    print("Retrieving context + generating answer...\n")

    result = retrieve_and_generate(question, lang)

    print("=== Retrieved context (top chunks) ===")
    for i, chunk in enumerate(result["retrieved_chunks"], start=1):
        print(f"  [{i}] {chunk[:150]}...")

    print("\n=== Generated answer ===")
    print(result["answer"])


if __name__ == "__main__":
    main()