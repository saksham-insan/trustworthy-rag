I'm building a final-year B.Tech project (Computer Science, Data Science specialization) and need help implementing it end-to-end. Here's the complete context — please read all of it before we start, since I'll be working with you across many sessions on this.

## Project Title
Trustworthy RAG: Reducing AI Hallucination in Regional Language Q&A Using Multi-Agent Verification

## What This Project Is
A Multi-Agent Retrieval-Augmented Generation (RAG) system that answers factual questions from a document knowledge base, specifically tested across English, Hindi, Kannada, and code-mixed "Kanglish" queries. A second LLM pass acts as a verifier agent, checking whether each generated answer is actually supported by the retrieved context before it's shown to the user.

## My Constraints (important — design everything around these)
- Hardware: CPU-only laptop, 12th Gen Intel i5, 8GB RAM, no GPU. Nothing in this project should require local model training or GPU inference.
- Budget: Must use free-tier tools only (free LLM APIs like Gemini/Groq, open-source embedding models, free vector databases).
- Time: A few hours a week, occasionally more when I have free time, over roughly 4 months total.
- I want clean, well-commented code I can actually understand and explain in a viva — not a black box.

## What This Project Is NOT Claiming (so you calibrate correctly)
- I am NOT inventing a new RAG architecture, a new embedding model, or a new verifier technique. RAG, verifier agents (Self-RAG, CRAG), and multilingual embeddings all already exist.
- My contribution is an empirical validation study: I'm the one actually testing this combination at real scale, on Kannada specifically, and running one experiment nobody else has run (see below).

## My Actual Research Contribution (the part that's genuinely mine)
1. **The core novelty:** A controlled ablation comparing a monolingual vector index (one language per database) vs. a shared multilingual vector index (all languages combined in one database) — measuring whether combining languages causes retrieval interference, per language.
2. Empirical validation at real scale: live LLM APIs (not offline/fallback stand-ins), a proper test set (~200-400 questions across languages), and Kannada actually tested with real numbers rather than assumed to work.
3. Measuring how much a verifier agent reduces hallucination specifically for Hindi/Kannada/Kanglish, where retrieval is already known to be weaker than English.

## Research Questions
- RQ1: How much does retrieval accuracy drop for Hindi, Kannada, and Kanglish compared to English?
- RQ2: Does a verifier agent reduce hallucination rate, and by how much, per language?
- RQ3: Does a shared multilingual index perform worse than separate monolingual indexes — for each language, including English?
- RQ4: What's the accuracy-vs-latency trade-off of adding the verifier step?

## System Architecture (the pipeline to build)
```
User query (English/Hindi/Kannada/Kanglish)
        ↓
Multilingual embedding model (converts query to vector)
        ↓
Vector index — TWO conditions to build and compare:
  (a) Monolingual: separate index per language
  (b) Multilingual: one shared index, all languages combined
        ↓
Retrieved chunks → LLM generator (answers in the query's language)
        ↓
Verifier agent (2nd LLM call — checks answer against retrieved context, flags/regenerates if unsupported)
        ↓
Final answer + full logging (query, retrieved chunks, answer, verifier verdict, latency) to SQLite for evaluation
```

## Planned Tech Stack
- Python 3.10+
- Embeddings: `sentence-transformers`, specifically `multilingual-e5-base` (runs fine on CPU for small corpora)
- Vector database: ChromaDB
- LLM APIs: Gemini and/or Groq (free tiers) — need both a generator call and a separate verifier call
- Backend: FastAPI
- Frontend: Streamlit (simple, just enough to demo query → answer)
- Logging/evaluation storage: SQLite
- Document handling: PyMuPDF or pdfplumber for PDF extraction, if source documents are PDFs

## Documents / Domain
[Fill this in when we start — e.g., "Christ University academic policy documents" or "a specific set of government scheme FAQs" — I haven't finalized which document set to use yet, help me decide based on ease of getting Hindi/Kannada/Kanglish test data for it.]

## Evaluation Plan
- Build a parallel test set: same ~50-100 questions, translated into English/Hindi/Kannada/Kanglish, with known correct answers and known correct source chunks.
- Metrics: retrieval precision@k per language per index condition; hallucination rate with vs. without verifier; response latency per condition.
- Log everything to SQLite so I can generate result tables/charts afterward for my paper.

## What I Need From You
Help me build this step by step, starting with environment setup and document ingestion, then the basic RAG pipeline, then the two index conditions, then the verifier agent, then the FastAPI + Streamlit wrapper, then the evaluation harness. Please:
- Write clean, commented Python I can actually explain in a viva.
- Flag anything that might not work well on 8GB RAM before I hit an out-of-memory error.
- Suggest free-tier API options at each step and note any rate-limit issues I should plan around.
- Keep me focused on getting a basic end-to-end version working fast, before adding complexity.

Let's start with environment setup and confirming the document domain — what should I do first?
