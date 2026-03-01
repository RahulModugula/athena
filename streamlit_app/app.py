import streamlit as st

st.set_page_config(
    page_title="Athena — Research Assistant",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

st.title("🦉 Athena")
st.subheader("RAG-Powered Research Assistant")

st.markdown("""
Athena lets you upload research documents, ask questions with hybrid retrieval,
and evaluate answer quality with RAGAS metrics.

**Features:**
- 📄 Upload PDFs, text files, and Markdown documents
- 🔍 Hybrid search (dense vectors + BM25) with cross-encoder reranking
- 💬 Streaming answers with source citations
- 📊 Evaluation dashboard with chunking strategy benchmarks

**Navigation:** Use the sidebar to switch between Upload, Search, and Evaluate.
""")

import httpx

try:
    with httpx.Client(timeout=3.0) as client:
        resp = client.get(f"{API_URL}/api/health")
        health = resp.json()
    st.success(f"API connected — {health['document_count']} documents loaded")
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", health["document_count"])
    col2.metric("Embedding Model", health["embedding_model"].split("/")[-1])
    col3.metric("LLM", health["llm_model"])
except Exception:
    st.warning("API not reachable — start the backend with `docker compose up`")
