import httpx
import pandas as pd
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Evaluate", page_icon="📊", layout="wide")
st.title("📊 Evaluation Dashboard")
st.markdown("Run RAGAS evaluations and compare chunking strategy performance.")

st.info("To run a full benchmark with RAGAS metrics, use the CLI:\n```\npython -m eval.runner --benchmark\n```")

with st.expander("Run Quick Evaluation via API"):
    with st.form("eval_form"):
        col1, col2 = st.columns(2)
        chunking = col1.selectbox("Chunking Strategy", ["recursive", "fixed", "semantic"])
        retrieval = col2.selectbox("Retrieval Strategy", ["hybrid", "dense", "bm25"])
        run_btn = st.form_submit_button("Run Evaluation", type="primary")

    if run_btn:
        with st.spinner("Running evaluation..."):
            try:
                with httpx.Client(timeout=300.0) as client:
                    resp = client.post(
                        f"{API_URL}/api/eval/run",
                        json={"chunking_strategy": chunking, "retrieval_strategy": retrieval},
                    )
                if resp.status_code == 200:
                    st.success("Evaluation recorded.")
                else:
                    st.error(resp.json().get("detail", "failed"))
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.subheader("Past Evaluation Runs")

try:
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{API_URL}/api/eval/results")
    runs = resp.json()

    if runs:
        rows = []
        for run in runs:
            m = run["metrics"]
            rows.append({
                "Date": run["created_at"][:10],
                "Chunking": run["chunking_strategy"],
                "Retrieval": run["retrieval_strategy"],
                "Faithfulness": m["faithfulness"],
                "Answer Relevance": m["answer_relevance"],
                "Context Precision": m["context_precision"],
                "Context Recall": m["context_recall"],
                "Samples": run["sample_count"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        if len(df) > 1:
            import plotly.express as px
            fig = px.bar(
                df,
                x="Chunking",
                y=["Faithfulness", "Answer Relevance", "Context Precision", "Context Recall"],
                barmode="group",
                title="Metrics by Chunking Strategy",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No evaluation runs yet.")
except Exception as e:
    st.error(f"Could not fetch results: {e}")
