import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.agents.llm import get_agent_llm
from app.agents.state import ResearchState

logger = structlog.get_logger()

SUPERVISOR_SYSTEM = """You are a research supervisor. Your job is to plan a thorough research strategy for answering a question.

Given the user's question, produce a concise research plan (2-4 sentences) that describes:
1. What information needs to be retrieved
2. What aspects to analyze
3. What claims need fact-checking

Keep the plan focused and actionable."""


class ResearchPlan(BaseModel):
    plan: str


async def supervisor_node(state: ResearchState) -> dict:
    logger.info("supervisor planning", question=state["question"][:80])
    llm = get_agent_llm()
    structured = llm.with_structured_output(ResearchPlan)

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content=f"Question: {state['question']}"),
    ]
    result: ResearchPlan = await structured.ainvoke(messages)  # type: ignore[assignment]

    return {
        "plan": result.plan,
        "iteration": state.get("iteration", 0) + 1,
    }
