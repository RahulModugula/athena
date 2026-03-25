import json

import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Search", page_icon="🔍", layout="wide")
st.title("🔍 Ask a Question")

question = st.text_area(
    "Your question",
    placeholder="What is the attention mechanism in transformers?",
    height=100,
)

col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
strategy = col1.selectbox("Retrieval Strategy", ["hybrid", "dense", "bm25"])
top_k = col2.slider("Number of sources", min_value=1, max_value=10, value=5)
deep_research = col3.toggle("Deep Research", value=False, help="Use multi-agent research pipeline")
ask = col4.button("Ask", type="primary", use_container_width=True)

if ask and question.strip():
    answer_placeholder = st.empty()
    sources_placeholder = st.empty()

    full_answer = ""
    sources = []

    if deep_research:
        with st.spinner("Running deep research pipeline..."):
            try:
                with httpx.Client(timeout=300.0) as client:
                    with client.stream(
                        "POST",
                        f"{API_URL}/api/research/stream",
                        json={"question": question, "max_iterations": 3},
                    ) as response:
                        for line in response.iter_lines():
                            if not line.startswith("data: "):
                                continue
                            event = json.loads(line[6:])
                            if event["type"] == "source":
                                sources.append(event["data"])
                            elif event["type"] == "chunk":
                                full_answer += event["data"]
                                answer_placeholder.markdown(f"**Answer:**\n\n{full_answer}▌")
                            elif event["type"] == "step":
                                st.caption(f"Step: {event['data']}")
                            elif event["type"] == "done":
                                answer_placeholder.markdown(f"**Answer:**\n\n{full_answer}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        with st.spinner("Retrieving and generating..."):
            try:
                with httpx.Client(timeout=120.0) as client:
                    with client.stream(
                        "POST",
                        f"{API_URL}/api/query/stream",
                        json={"question": question, "strategy": strategy, "top_k": top_k},
                    ) as response:
                        for line in response.iter_lines():
                            if not line.startswith("data: "):
                                continue
                            event = json.loads(line[6:])
                            if event["type"] == "source":
                                sources.append(event["data"])
                            elif event["type"] == "chunk":
                                full_answer += event["data"]
                                answer_placeholder.markdown(f"**Answer:**\n\n{full_answer}▌")
                            elif event["type"] == "done":
                                answer_placeholder.markdown(f"**Answer:**\n\n{full_answer}")
            except Exception as e:
                st.error(f"Error: {e}")

    if sources:
        st.divider()
        st.subheader(f"Sources ({len(sources)})")
        for i, source in enumerate(sources, 1):
            with st.expander(
                f"[{i}] {source.get('document_name', 'unknown')} — chunk {source.get('chunk_index', '?')} (score: {source.get('score', 0):.4f})"
            ):
                st.text(source.get("content", ""))
elif ask:
    st.warning("Please enter a question.")
