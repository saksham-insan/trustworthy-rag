"""
Step 7b: Streamlit frontend.

A simple demo UI: type a question, pick a language and index condition,
see the retrieved context, generated answer, and verifier verdict.
Calls the FastAPI backend over HTTP (so make sure that's running first).

Usage:
  Terminal 1: uvicorn src.api:app --reload --port 8000
  Terminal 2: streamlit run src/app.py
"""

import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="Trustworthy RAG Demo", layout="wide")

st.title("Trustworthy RAG: Multi-Agent Verification Demo")
st.caption("Ask a question about the loaded government scheme documents (English / Hindi / Bengali).")

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    question = st.text_input("Your question", placeholder="e.g. What is the eligibility for the scholarship?")
with col2:
    language = st.selectbox("Language", ["en", "hi", "bn"], format_func=lambda l: {"en": "English", "hi": "Hindi", "bn": "Bengali"}[l])
with col3:
    index_condition = st.selectbox("Index condition", ["mono", "multi"], format_func=lambda c: {"mono": "Monolingual", "multi": "Multilingual"}[c])

verifier_enabled = st.checkbox(
    "Enable verifier agent",
    value=True,
    help="Turn off to see the generator's raw answer with no hallucination check — useful for comparing with/without the verifier (RQ2).",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving context, generating answer" + (", verifying..." if verifier_enabled else "...")):
        try:
            response = requests.post(
                API_URL,
                json={
                    "question": question,
                    "language": language,
                    "index_condition": index_condition,
                    "verifier_enabled": verifier_enabled,
                },
                timeout=180,  # generous: retries on rate limits can add 10-30s+ per stage
            )
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                st.error(f"Request failed ({response.status_code}): {detail}")
                st.stop()
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Can't reach the API. Make sure it's running: `uvicorn src.api:app --reload --port 8000`")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timed out. The APIs may be under heavy rate-limiting right now — try again in a minute.")
            st.stop()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    st.subheader("Answer")
    st.write(result["answer"])

    verdict = result["verifier_verdict"]
    verdict_color = {
        "SUPPORTED": "green", "UNSUPPORTED": "red", "PARTIALLY_SUPPORTED": "orange", "NOT_VERIFIED": "gray",
    }.get(verdict, "gray")
    st.markdown(f"**Verifier verdict:** :{verdict_color}[{verdict}]")
    st.caption(result["verifier_explanation"])

    latency_col1, latency_col2, latency_col3, latency_col4 = st.columns(4)
    latency_col1.metric("Retrieval", f"{result['retrieval_latency_ms']:.0f} ms")
    latency_col2.metric("Generation", f"{result['generation_latency_ms']:.0f} ms")
    latency_col3.metric("Verification", f"{result['verification_latency_ms']:.0f} ms")
    latency_col4.metric("Total", f"{result['total_latency_ms']:.0f} ms")

    with st.expander(f"Retrieved context ({len(result['retrieved_chunks'])} chunks)"):
        for i, (chunk, lang) in enumerate(zip(result["retrieved_chunks"], result["retrieved_languages"]), start=1):
            st.markdown(f"**[{i}] language: `{lang}`**")
            st.text(chunk)
            st.divider()
else:
    st.info("Enter a question above and click Ask.")