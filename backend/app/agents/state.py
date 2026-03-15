from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    question: str
    plan: str
    retrieved_chunks: list[dict[str, Any]]
    analysis: str
    fact_check_results: list[dict[str, Any]]
    draft_answer: str
    final_answer: str
    sources: list[dict[str, Any]]
    messages: Annotated[list[AnyMessage], add_messages]
    iteration: int
    max_iterations: int
