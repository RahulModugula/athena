import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_agent_llm
from app.agents.state import ResearchState
from app.generation.prompts import format_context

logger = structlog.get_logger()

ANALYST_SYSTEM = """You are a research analyst. Given a question, a research plan, and retrieved source passages, produce a structured analysis.

Your analysis must:
- Identify key findings relevant to the question
- Note any contradictions or gaps in the sources
- Cite sources using [Source N] notation
- Be factual and grounded only in the provided passages
- Be 150-300 words"""


async def analyst_node(state: ResearchState) -> dict:
    question = state["question"]
    chunks = state.get("retrieved_chunks", [])
    plan = state.get("plan", "")
    logger.info("analyst analyzing", chunks=len(chunks))

    if not chunks:
        return {"analysis": "No relevant sources found to analyze."}

    context = format_context(chunks)
    llm = get_agent_llm()
    messages = [
        SystemMessage(content=ANALYST_SYSTEM),
        HumanMessage(
            content=f"Question: {question}\nPlan: {plan}\n\nSources:\n{context}"
        ),
    ]
    response = await llm.ainvoke(messages)
    analysis = str(response.content)
    logger.info("analyst done", analysis_length=len(analysis))
    return {"analysis": analysis}
