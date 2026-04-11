"""Source viewer component for displaying document spans with highlights."""

import streamlit as st


def render_highlighted_span(
    document_text: str,
    start_offset: int,
    end_offset: int,
    context_chars: int = 200,
) -> str:
    """Render a document span with context and highlighting.

    Args:
        document_text: Full document text
        start_offset: Start character offset
        end_offset: End character offset
        context_chars: Number of context characters before/after span

    Returns:
        HTML string with highlighted span
    """
    # Clamp offsets
    context_start = max(0, start_offset - context_chars)
    context_end = min(len(document_text), end_offset + context_chars)

    before = document_text[context_start:start_offset]
    span = document_text[start_offset:end_offset]
    after = document_text[end_offset:context_end]

    # Build HTML
    html = f"""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; font-family: monospace; line-height: 1.6;">
        {before}<span style="background-color: #FFE082; font-weight: bold; padding: 0.2rem 0.4rem; border-radius: 0.25rem;">{span}</span>{after}
    </div>
    """
    return html


def display_source_snippet(
    chunk_content: str,
    document_name: str,
    start_offset: int | None = None,
    end_offset: int | None = None,
) -> None:
    """Display a source chunk with optional span highlighting.

    Args:
        chunk_content: The chunk content to display
        document_name: Name of the source document
        start_offset: Start offset for highlighting (optional)
        end_offset: End offset for highlighting (optional)
    """
    st.markdown(f"**Source:** {document_name}")

    if start_offset is not None and end_offset is not None:
        # If we have offsets, try to highlight the span
        try:
            html = render_highlighted_span(chunk_content, start_offset, end_offset)
            st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not highlight span: {e}")
            st.text(chunk_content)
    else:
        # Just display the chunk as-is
        st.text(chunk_content)


def display_multiple_sources(sources: list[dict]) -> None:
    """Display multiple source chunks in expandable sections.

    Args:
        sources: List of source dicts with 'content', 'document_name', etc
    """
    st.subheader(f"Sources ({len(sources)})")

    for i, source in enumerate(sources, 1):
        doc_name = source.get("document_name", "unknown")
        chunk_index = source.get("chunk_index", 0)
        score = source.get("score", 0.0)

        with st.expander(f"[{i}] {doc_name} (chunk {chunk_index}, score: {score:.4f})"):
            display_source_snippet(
                chunk_content=source.get("content", ""),
                document_name=doc_name,
                start_offset=source.get("start_offset"),
                end_offset=source.get("end_offset"),
            )
