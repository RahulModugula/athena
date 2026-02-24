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


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("document_name", "unknown")
        content = chunk.get("content", "")
        parts.append(f"[Source {i}] ({source})\n{content}")
    return "\n\n---\n\n".join(parts)
