"""
Shared configuration for the Trustworthy RAG project.
Keep every "magic number" and model name here so the whole pipeline is
tweakable from one place — makes it easy to explain design choices in a viva.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHROMA_MONO_DIR = PROJECT_ROOT / "chroma_db_mono"      # monolingual index condition
CHROMA_MULTI_DIR = PROJECT_ROOT / "chroma_db_multi"    # shared multilingual index condition
EVAL_DB_PATH = PROJECT_ROOT / "eval" / "logs.sqlite"

# --- Supported languages ---
# "benglish" = Bengali-English code-mixed queries (our own label, not a real ISO code).
# It only appears in test QUERIES, never in the source knowledge base.
SUPPORTED_LANGUAGES = ["en", "hi", "bn", "benglish"]

# --- Chunking ---
# Small-ish chunks: multilingual-e5-base has a 512 token context; we chunk
# well under that so retrieval granularity stays useful (whole-paragraph
# answers, not whole-document dumps).
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 150

# --- Embedding model ---
# multilingual-e5-base: ~278M params, ~1.1GB fp32, runs fine on CPU for a
# corpus of a few hundred to a few thousand chunks. Supports 100+ languages
# including Hindi and Bengali.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_BATCH_SIZE = 16  # keep small on 8GB RAM; raise cautiously if headroom allows

# NOTE: e5 models expect a "query: " / "passage: " prefix on inputs — this is
# NOT optional, it's how the model was trained. We handle this in embed_index.py.

# --- Retrieval ---
TOP_K = 5

# --- LLM generation ---
GENERATION_MAX_TOKENS = 512
GENERATION_TEMPERATURE = 0.2  # low temperature: we want grounded, not creative, answers

# --- LLM model names (current as of this project's setup — verify periodically,
#     free-tier providers retire/rename models fairly often) ---
GEMINI_MODEL_NAME = "gemini-flash-lite-latest"
GROQ_MODEL_NAME = "openai/gpt-oss-20b"