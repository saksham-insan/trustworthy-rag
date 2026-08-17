# Trustworthy RAG: Reducing AI Hallucination in Regional Language Q&A Using Multi-Agent Verification

Final-year B.Tech project (CS, Data Science). See `PROJECT_BRIEF.md` for full context.

## Status
🟢 Step 0: Environment + repo scaffolding — done
🟡 Step 1: Document domain decision — proposed below, confirm before ingesting
⬜ Step 2: Document ingestion + chunking
⬜ Step 3: Basic RAG pipeline (single index, English only)
⬜ Step 4: Monolingual vs multilingual index conditions
⬜ Step 5: Verifier agent
⬜ Step 6: FastAPI + Streamlit wrapper
⬜ Step 7: Evaluation harness + SQLite logging
⬜ Step 8: Parallel multilingual test set + results

---

## Document domain — recommendation

You need a corpus that genuinely exists in English, Hindi, and Kannada natively (not just
machine-translated by you), because part of RQ1/RQ3 is about *real* multilingual retrieval,
not translation quality.

**Recommended: Government scheme / citizen-service FAQs**, specifically:

- **Primary source: [myScheme](https://www.myscheme.gov.in)** — central govt portal, most
  scheme pages have English + Hindi content, and a decent subset have Kannada. Good density
  of factual, extractable QA-style content (eligibility, benefits, documents required).
- **Backup / supplement: Karnataka govt department sites (Seva Sindhu, DBT Karnataka,
  Department of Kannada and Culture)** — for guaranteed Kannada-native documents, since
  myScheme's Kannada coverage may be uneven.
- **Fallback if native trilingual PDFs are too hard to find in enough volume:** pick ~15-20
  solid schemes/policies in English, get official Hindi versions where they exist, and for
  Kannada use a mix of native department content + Google Translate API (free tier) for the
  rest — but log clearly in your report which chunks are native vs. translated, since that's
  a legitimate caveat to discuss in your viva, not something to hide.

**Why this over "Christ University policies":** university policy PDFs are almost always
English-only, so you'd have to translate everything yourself, which quietly turns your
"real multilingual retrieval" experiment into "translation quality experiment." Government
scheme content is one of the few domains in India with a realistic chance of natively
existing in all three languages at the volume you need (a few hundred chunks per language).

**Action needed from you:** browse myScheme + 2-3 Karnataka dept sites, download ~15-25 PDFs/
pages per language (start with 5-10 to test the pipeline first), drop them in `data/raw/`.
We'll adjust scope once we see real document availability — this is a "best guess, verify
early" decision, not a locked-in one.

---

## Environment setup

```bash
# 1. Clone / init repo (see Git section below if starting fresh)
cd trustworthy-rag

# 2. Create virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install CPU-only torch FIRST (avoids accidentally pulling a huge CUDA build)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# 4. Install the rest
pip install -r requirements.txt

# 5. Copy env template and fill in your free API keys
cp .env.example .env
# edit .env: add GEMINI_API_KEY and/or GROQ_API_KEY
```

**Getting free API keys:**
- Gemini: https://aistudio.google.com/apikey (free tier, rate-limited per minute/day)
- Groq: https://console.groq.com/keys (free tier, very fast inference, good for verifier calls)

**8GB RAM notes (read before running anything):**
- `multilingual-e5-base` (~1.1GB in fp32) is fine to load once and keep resident — don't
  reload it per request in FastAPI, load it once at startup.
- Embed documents in batches of 16-32, not all at once, during ingestion — see `src/ingest.py`.
- Don't run Streamlit + FastAPI + a Jupyter kernel + the embedding model all at once while
  also having 20 Chrome tabs open. Close what you don't need.
- ChromaDB persists to disk by default — fine for our corpus size (hundreds to low
  thousands of chunks), no in-memory scaling concern.

---

## Git setup

If this is a brand new repo:

```bash
cd trustworthy-rag
git init
git add .
git commit -m "chore: project scaffolding, requirements, README"

# Create an empty repo on GitHub first (via github.com), then:
git branch -M main
git remote add origin https://github.com/<your-username>/trustworthy-rag.git
git push -u origin main
```

Recommended branch habit for a solo project (keeps history readable for your viva/report):

```bash
git checkout -b feat/ingestion
# ... do work ...
git add -A
git commit -m "feat: PDF ingestion + chunking pipeline"
git checkout main
git merge feat/ingestion
git push
```

Commit early and often — your report/viva benefits from a clean commit history showing
incremental progress (ingestion → basic RAG → dual-index → verifier → eval), and it's
evidence the work is genuinely yours.

---

## Project structure

```
trustworthy-rag/
├── src/
│   ├── ingest.py          # PDF/text extraction + chunking (Step 2 — included now)
│   ├── embed_index.py     # embedding + ChromaDB indexing (Step 2/4 — next)
│   ├── generator.py       # LLM answer generation (Step 3 — next)
│   ├── verifier.py        # verifier agent (Step 5)
│   ├── pipeline.py        # ties retrieval + generation + verification together
│   └── config.py          # shared config (model names, paths, constants)
├── data/
│   ├── raw/                # source PDFs/docs, gitignored (keep .gitkeep)
│   └── processed/          # chunked JSON/CSV output of ingest.py
├── eval/                   # test sets, SQLite logging, result notebooks
├── notebooks/               # exploratory analysis for charts/tables in your paper
├── requirements.txt
├── .env.example
└── README.md
```

## Next step
Once you've dropped a handful of sample PDFs into `data/raw/`, run `src/ingest.py` (see
docstring at top of file) to test extraction + chunking end-to-end before we scale up.
