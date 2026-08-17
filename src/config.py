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

# --- Supported languages (extend later if needed) ---
# Keep language codes consistent everywhere: ISO-639-1 where possible,
# "kanglish" is our own label since it isn't a real ISO code.
SUPPORTED_LANGUAGES = ["en", "hi", "kn", "kanglish"]

# --- Chunking ---
# Small-ish chunks: multilingual-e5-base has a 512 token context; we chunk
# well under that so retrieval granularity stays useful (whole-paragraph
# answers, not whole-document dumps).
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 150

# --- Embedding model ---
# multilingual-e5-base: ~278M params, ~1.1GB fp32, runs fine on CPU for a
# corpus of a few hundred to a few thousand chunks. Supports 100+ languages
# including Hindi and Kannada.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_BATCH_SIZE = 16  # keep small on 8GB RAM; raise cautiously if headroom allows

# NOTE: e5 models expect a "query: " / "passage: " prefix on inputs — this is
# NOT optional, it's how the model was trained. We handle this in embed_index.py.

# --- Retrieval ---
TOP_K = 5

# --- LLM generation ---
GENERATION_MAX_TOKENS = 512
GENERATION_TEMPERATURE = 0.2  # low temperature: we want grounded, not creative, answers
