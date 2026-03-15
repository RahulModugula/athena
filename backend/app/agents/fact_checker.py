import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.agents.llm import get_agent_llm
from app.agents.state import ResearchState

logger = structlog.get_logger()

FACT_CHECK_SYSTEM = """You are a fact-checker. Given an analysis and the original source passages, verify the key claims.

For each significant claim in the analysis:
1. Determine if it is supported by the source passages
2. Rate your confidence (0.0 to 1.0)
3. Note supporting evidence or flag unsupported claims

Return a list of fact-check results."""


class FactCheckItem(BaseModel):
    claim: str
    supported: bool
    confidence: float
    evidence: list[str]


class FactCheckResults(BaseModel):
    results: list[FactCheckItem]


async def fact_checker_node(state: ResearchState) -> dict:
    analysis = state.get("analysis", "")
    chunks = state.get("retrieved_chunks", [])
    logger.info("fact_checker checking", analysis_length=len(analysis))

    if not analysis or not chunks:
        return {"fact_check_results": []}

    sources_text = "\n\n".join(
        f"[Source {i + 1}]: {c.get('content', '')[:400]}"
        for i, c in enumerate(chunks[:8])
    )
    llm = get_agent_llm()
    structured = llm.with_structured_output(FactCheckResults)
    messages = [
        SystemMessage(content=FACT_CHECK_SYSTEM),
        HumanMessage(
            content=f"Analysis to verify:\n{analysis}\n\nSource passages:\n{sources_text}"
        ),
    ]
    result: FactCheckResults = await structured.ainvoke(messages)
    items = [item.model_dump() for item in result.results]
    logger.info("fact_checker done", claims=len(items))
    return {"fact_check_results": items}
