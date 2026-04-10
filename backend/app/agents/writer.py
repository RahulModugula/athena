import json
import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_agent_llm
from app.agents.state import ResearchState
from app.generation.prompts import format_context
from app.verification.models import VerifiedAnswerDraft

logger = structlog.get_logger()

WRITER_SYSTEM = """You are a research writer. Synthesize a clear, accurate, and well-cited answer.

IMPORTANT: Respond with ONLY valid JSON in this exact format:
{
  "sentences": [
    {
      "text": "The claim or sentence.",
      "citations": [
        {"chunk_id": "uuid-of-chunk", "start": 100, "end": 150}
      ]
    }
  ]
}

For each sentence:
1. Write a clear claim
2. Cite the exact chunks and character offsets that support it
3. Use only claims supported by the source passages
4. Each offset range must point to real text in the cited chunk

Guidelines:
- Only write sentences supported by retrieved chunks
- Use true chunk IDs and valid offsets
- Aim for 3-5 sentences total
- Never include commentary or explanation outside the JSON"""


async def writer_node(state: ResearchState) -> dict:
    question = state["question"]
    analysis = state.get("analysis", "")
    fact_checks = state.get("fact_check_results", [])
    chunks = state.get("retrieved_chunks", [])
    logger.info("writer synthesizing", chunks=len(chunks))

    context = format_context(chunks) if chunks else "No sources available."

    # Build chunk reference for writer
    chunk_refs = "\n".join(
        f"Chunk {i}: ID={c.get('chunk_id', '')}, content={c.get('content', '')[:100]}..."
        for i, c in enumerate(chunks[:10])
    )

    llm = get_agent_llm()
    structured_llm = llm.with_structured_output(VerifiedAnswerDraft)

    messages = [
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"Analysis:\n{analysis}\n\n"
                f"Available chunks for citation:\n{chunk_refs}"
            )
        ),
    ]

    try:
        response = await structured_llm.ainvoke(messages)
        draft = response
        # Convert back to JSON for storage
        final_answer = json.dumps(draft.model_dump())
    except Exception as e:
        logger.warning("structured output failed, falling back to freeform", error=str(e))
        # Fallback: unstructured answer
        unstructured_llm = llm
        messages_fallback = [
            SystemMessage(content="You are a research writer. Write a concise answer."),
            HumanMessage(
                content=(
                    f"Question: {question}\n\n"
                    f"Analysis:\n{analysis}\n\n"
                    f"Sources:\n{context}"
                )
            ),
        ]
        response = await unstructured_llm.ainvoke(messages_fallback)
        final_answer = str(response.content)

    sources = [
        {
            "chunk_id": str(c.get("chunk_id", "")),
            "content": c.get("content", ""),
            "document_name": c.get("document_name", ""),
            "chunk_index": c.get("chunk_index", 0),
            "score": c.get("score", 0.0),
        }
        for c in chunks
    ]

    logger.info("writer done", answer_length=len(final_answer))
    return {"final_answer": final_answer, "sources": sources}
