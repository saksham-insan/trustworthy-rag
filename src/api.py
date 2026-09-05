"""
Step 7a: FastAPI backend.

Wraps the full pipeline (retrieve -> generate -> verify -> log) as a web
API, so the Streamlit frontend (and anything else later) can call it over
HTTP instead of importing Python functions directly. This also mirrors a
realistic production setup, which is worth mentioning in your report.

Usage:
  uvicorn src.api:app --reload --port 8000

  Then test with:
  http://localhost:8000/docs   (interactive API docs, auto-generated)
"""

import truststore
truststore.inject_into_ssl()

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import run_pipeline

app = FastAPI(title="Trustworthy RAG API", version="0.1.0")


class QueryRequest(BaseModel):
    question: str
    language: str          # "en" | "hi" | "bn"
    index_condition: str = "mono"   # "mono" | "multi"
    verifier_enabled: bool = True   # set False to test the pipeline WITHOUT the verifier (RQ2 ablation)


class QueryResponse(BaseModel):
    question: str
    language: str
    index_condition: str
    verifier_enabled: bool
    answer: str
    verifier_verdict: str
    verifier_explanation: str
    retrieved_chunks: list[str]
    retrieved_languages: list[str]
    retrieval_latency_ms: float
    generation_latency_ms: float
    verification_latency_ms: float
    total_latency_ms: float


@app.get("/")
def root():
    return {"status": "ok", "message": "Trustworthy RAG API is running. See /docs for usage."}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    if request.language not in ("en", "hi", "bn"):
        raise HTTPException(status_code=400, detail="language must be one of: en, hi, bn")
    if request.index_condition not in ("mono", "multi"):
        raise HTTPException(status_code=400, detail="index_condition must be 'mono' or 'multi'")

    try:
        record = run_pipeline(request.question, request.language, request.index_condition, request.verifier_enabled)
    except RuntimeError as e:
        # RuntimeError here means a setup problem (e.g. index not built yet) —
        # that's a client-fixable issue, not a server crash, so 400 not 500.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    return QueryResponse(
        question=record["question"],
        language=record["language"],
        index_condition=record["index_condition"],
        verifier_enabled=record["verifier_enabled"],
        answer=record["answer"],
        verifier_verdict=record["verifier_verdict"],
        verifier_explanation=record["verifier_explanation"],
        retrieved_chunks=record["retrieved_chunks"],
        retrieved_languages=record["retrieved_languages"],
        retrieval_latency_ms=record["retrieval_latency_ms"],
        generation_latency_ms=record["generation_latency_ms"],
        verification_latency_ms=record["verification_latency_ms"],
        total_latency_ms=record["total_latency_ms"],
    )