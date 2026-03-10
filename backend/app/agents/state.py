from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    question: str
    plan: str
    retrieved_chunks: list[dict]
    analysis: str
    fact_check_results: list[dict]
    draft_answer: str
    final_answer: str
    sources: list[dict]
    messages: Annotated[list[AnyMessage], add_messages]
    iteration: int
    max_iterations: int
