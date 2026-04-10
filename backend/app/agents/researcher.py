import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.agents.llm import get_agent_llm
from app.agents.state import ResearchState

logger = structlog.get_logger()

DECOMPOSE_SYSTEM = """You are a research assistant. Given a question and a research plan, decompose the question into 2-3 targeted sub-queries that together cover the full scope of the research plan.

Return only the sub-queries as a JSON list of strings."""


class SubQueries(BaseModel):
    queries: list[str]


async def researcher_node(state: ResearchState) -> dict:
    question = state["question"]
    plan = state.get("plan", "")
    weak_claims = state.get("weak_claims", [])
    iteration = state.get("iteration", 0)
    logger.info("researcher retrieving", question=question[:80], iteration=iteration, weak_claims=len(weak_claims))

    llm = get_agent_llm()
    structured = llm.with_structured_output(SubQueries)

    # If retrying, bias decomposition toward weak claims
    if weak_claims and iteration > 0:
        decompose_prompt = f"""You are a research assistant. The previous retrieval failed to support these claims:

{chr(10).join(f"- {claim}" for claim in weak_claims[:3])}

Given the original question and what failed, decompose into 2-3 targeted sub-queries that specifically search for evidence to support those weak claims.

Return only the sub-queries as a JSON list of strings."""
        messages = [
            SystemMessage(content=decompose_prompt),
            HumanMessage(content=f"Original question: {question}"),
        ]
    else:
        messages = [
            SystemMessage(content=DECOMPOSE_SYSTEM),
            HumanMessage(content=f"Question: {question}\nPlan: {plan}"),
        ]

    decomposed: SubQueries = await structured.ainvoke(messages)

    # Lazy import to avoid circular deps at module load time
    from app.services.retrieval_service import RetrievalService

    service: RetrievalService | None = state.get("_retrieval_service")

    all_chunks: list[dict] = []
    seen_ids: set[str] = set()

    queries = decomposed.queries or [question]
    for sub_query in queries[:3]:
        if service is not None:
            chunks = await service.retrieve(sub_query)
        else:
            chunks = []
        for chunk in chunks:
            cid = str(chunk.get("chunk_id", ""))
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(chunk)

    graph_ctx = ""
    graph_store = state.get("_graph_store")
    if graph_store is not None:
        from app.retrieval.graph_search import graph_context_search

        graph_ctx = await graph_context_search(question, graph_store)

    logger.info("researcher done", chunks=len(all_chunks), sub_queries=len(queries))
    return {
        "retrieved_chunks": all_chunks,
        "graph_context": graph_ctx,
        "iteration": iteration + 1,
    }
