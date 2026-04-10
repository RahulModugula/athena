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
    graph_context: str
    # Verification fields (Pillar 3-4)
    verified_sentences: list[dict[str, Any]]  # VerifiedSentence objects
    trust_score: float  # Overall answer trust score
    verification_passed: bool
    weak_claims: list[str]  # Claims to retry on if verification fails
    _retrieval_service: Any  # Passed by routes for retry logic
    _graph_store: Any  # Optional graph store for entity retrieval
