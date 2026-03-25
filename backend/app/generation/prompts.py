from langchain_core.prompts import ChatPromptTemplate

RAG_SYSTEM = """You are a precise research assistant. Answer the question using ONLY the context provided below.

Rules:
- If the context does not contain enough information, say "I don't have enough information in the provided documents to answer this."
- Cite your sources by referencing [Source N] where N is the source number.
- Be concise and accurate. Do not speculate beyond the context.
- If multiple sources support a point, cite all of them.

Context:
{context}"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM),
    ("human", "{question}"),
])


def lost_in_middle_reorder(chunks: list[dict]) -> list[dict]:
    """Reorder chunks so the most relevant appear at the start and end of context.

    LLMs suffer from a "lost in the middle" attention bias — information placed
    in the middle of a long context window is recalled less reliably than text
    at the start or end. This U-shaped ordering interleaves even-indexed chunks
    (high relevance) toward the front and odd-indexed chunks toward the back,
    so the most important evidence always brackets the context window.

    See: Liu et al., "Lost in the Middle" (arXiv 2307.03172).
    """
    if len(chunks) <= 2:
        return chunks
    result: list[dict] = []
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            result.insert(0, chunk)  # even-indexed → prepend (highest relevance first)
        else:
            result.append(chunk)     # odd-indexed → append (second-highest at end)
    return result


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block.

    Applies lost-in-the-middle reordering before formatting so the most
    relevant chunks anchor the beginning and end of the context window.
    """
    ordered = lost_in_middle_reorder(chunks)
    parts = []
    for i, chunk in enumerate(ordered, 1):
        source = chunk.get("document_name", "unknown")
        content = chunk.get("content", "")
        parts.append(f"[Source {i}] ({source})\n{content}")
    return "\n\n---\n\n".join(parts)
