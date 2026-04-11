"""Verified Query page: trustworthy RAG with citation verification."""

import json

import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Verified Query", page_icon="✅", layout="wide")
st.title("✅ Verified Query")
st.markdown(
    "Ask questions with **sentence-level trust scores** and verified citations."
)

question = st.text_area(
    "Your question",
    placeholder="What is the termination notice period in this contract?",
    height=100,
)

col1, col2, col3 = st.columns([2, 2, 1])
max_iterations = col1.slider("Max retry iterations", min_value=1, max_value=3, value=2)
show_trace = col2.toggle("Show verification trace", value=False)
ask = col3.button("Ask", type="primary", use_container_width=True)


def _render_trust_gauge(trust_score: float) -> str:
    """Render a simple Unicode trust gauge."""
    bars = int(trust_score * 10)
    filled = "█" * bars
    empty = "░" * (10 - bars)
    color = "🟢" if trust_score >= 0.85 else "🟡" if trust_score >= 0.6 else "🔴"
    return f"{color} {filled}{empty} {trust_score:.2%}"


def _render_sentence_with_trust(
    sentence_text: str, trust_score: float, status: str
) -> str:
    """Render a sentence with color-coded trust indicator."""
    if status == "SUPPORTED":
        color = "🟢"
    elif status == "PARTIAL":
        color = "🟡"
    elif status == "UNSUPPORTED":
        color = "🔴"
    else:  # CONTRADICTED
        color = "⛔"

    return f"{color} {sentence_text}\n\n*Trust: {trust_score:.1%}*"


if ask and question.strip():
    answer_placeholder = st.empty()
    status_placeholder = st.empty()
    trust_placeholder = st.empty()
    sentences_placeholder = st.empty()
    trace_placeholder = st.empty()

    with st.spinner("Running verified research pipeline..."):
        try:
            with httpx.Client(timeout=300.0) as client:
                with client.stream(
                    "POST",
                    f"{API_URL}/api/research/stream",
                    json={"question": question, "max_iterations": max_iterations},
                ) as response:
                    verified_data = None
                    final_answer = None

                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])

                        if event["type"] == "agent_start":
                            status_placeholder.info(f"Running {event['agent']}...")

                        elif event["type"] == "verification":
                            verified_data = event["data"]
                            trust_score = verified_data.get("trust_score", 0.0)
                            trust_placeholder.markdown(
                                f"### Overall Trust: {_render_trust_gauge(trust_score)}"
                            )

                        elif event["type"] == "done":
                            status_placeholder.empty()

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # Render verified sentences with trust indicators
    if verified_data and verified_data.get("verified_sentences"):
        st.divider()
        st.subheader("Verified Answer")

        sentences = verified_data["verified_sentences"]

        # Create tabs for different views
        tab1, tab2 = st.tabs(["Verified Sentences", "Verification Details"])

        with tab1:
            for i, sent in enumerate(sentences):
                sent_text = sent.get("text", "")
                trust = sent.get("trust_score", 0.0)
                status = sent.get("support_status", "UNSUPPORTED")

                with st.expander(
                    _render_sentence_with_trust(sent_text, trust, status),
                    expanded=False,
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Trust Score", f"{trust:.1%}")
                        st.metric("Support Status", status)

                    with col2:
                        st.metric("NLI Score", f"{sent.get('nli_score', 0.0):.3f}")
                        st.metric(
                            "Lexical Overlap",
                            f"{sent.get('lexical_overlap', 0.0):.1%}",
                        )

                    # Show citations
                    citations = sent.get("citations", [])
                    if citations:
                        st.markdown("**Citations:**")
                        for cit in citations:
                            st.code(
                                f"chunk_id: {cit.get('chunk_id')}\n"
                                f"span: {cit.get('start')}-{cit.get('end')}",
                                language="json",
                            )

        with tab2:
            if show_trace:
                st.subheader("Verification Trace")

                # Overall metrics
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Overall Trust", f"{verified_data.get('overall_trust_score', 0.0):.1%}"
                )
                col2.metric(
                    "Support Status", verified_data.get("overall_support_status", "?")
                )
                col3.metric(
                    "Verified",
                    "✅" if verified_data.get("verification_passed") else "❌",
                )

                # Weak claims
                weak_claims = verified_data.get("weak_claims", [])
                if weak_claims:
                    st.warning(f"⚠️ {len(weak_claims)} claims lack strong support:")
                    for claim in weak_claims:
                        st.caption(f"• {claim}")

                # Per-sentence details
                st.markdown("**Per-Sentence Breakdown:**")
                for i, sent in enumerate(sentences, 1):
                    st.json(
                        {
                            f"Sentence {i}": {
                                "text": sent.get("text", "")[:100] + "...",
                                "trust_score": f"{sent.get('trust_score', 0.0):.3f}",
                                "nli_score": f"{sent.get('nli_score', 0.0):.3f}",
                                "lexical_overlap": f"{sent.get('lexical_overlap', 0.0):.3f}",
                                "status": sent.get("support_status"),
                            }
                        }
                    )
            else:
                st.info("Toggle 'Show verification trace' above to see detailed scores.")

elif ask:
    st.warning("Please enter a question.")
