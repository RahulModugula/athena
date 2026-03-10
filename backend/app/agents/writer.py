import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_agent_llm
from app.agents.state import ResearchState
from app.generation.prompts import format_context

logger = structlog.get_logger()

WRITER_SYSTEM = """You are a research writer. Synthesize a clear, accurate, and well-cited answer from:
- The original question
- The analyst's findings
- The fact-checker's verification results
- The original source passages

Guidelines:
- Use [Source N] citations for every factual claim
- Only state things supported by the sources
- Flag any claims the fact-checker marked as unsupported
- Write in clear, professional prose
- Aim for 150-300 words"""


async def writer_node(state: ResearchState) -> dict:
    question = state["question"]
    analysis = state.get("analysis", "")
    fact_checks = state.get("fact_check_results", [])
    chunks = state.get("retrieved_chunks", [])
    logger.info("writer synthesizing", chunks=len(chunks))

    context = format_context(chunks) if chunks else "No sources available."

    unsupported = [
        fc["claim"] for fc in fact_checks if not fc.get("supported", True)
    ]
    unsupported_note = ""
    if unsupported:
        unsupported_note = "\n\nNote — these claims could NOT be verified in the sources:\n" + "\n".join(
            f"- {c}" for c in unsupported
        )

    llm = get_agent_llm()
    messages = [
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"Analysis:\n{analysis}{unsupported_note}\n\n"
                f"Sources:\n{context}"
            )
        ),
    ]
    response = await llm.ainvoke(messages)
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
