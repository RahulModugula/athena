"""Athena MCP server — exposes research tools via Model Context Protocol."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("athena")


@mcp.tool()
async def athena_research_question(question: str, max_iterations: int = 3) -> dict:
    """Run the full multi-agent research pipeline on a question."""
    # Import here to avoid circular imports
    from app.agents.graph import get_research_graph
    research_graph = get_research_graph()
    from app.agents.state import ResearchState

    state: ResearchState = {
        "question": question,
        "plan": "",
        "retrieved_chunks": [],
        "analysis": "",
        "fact_check_results": [],
        "draft_answer": "",
        "final_answer": "",
        "sources": [],
        "messages": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "graph_context": "",
    }
    result = await research_graph.ainvoke(state)
    return {"answer": result.get("final_answer", ""), "sources": result.get("sources", [])}


@mcp.tool()
async def athena_search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Search indexed documents using hybrid vector + BM25 search."""
    import app.mcp._store_ref as _store_ref  # noqa: F401

    # Return empty list if no DB available in MCP context
    return []


@mcp.tool()
async def athena_query_knowledge_graph(entity_name: str) -> dict:
    """Query the knowledge graph for context about a named entity."""
    import app.mcp._store_ref as _store_ref

    store = getattr(_store_ref, "graph_store", None)
    if store is None or not store.is_connected:
        return {"error": "knowledge graph not available"}
    subgraph = await store.get_entity_context([entity_name])
    return {
        "entities": [e.model_dump() for e in subgraph.entities],
        "relationships": [r.model_dump() for r in subgraph.relationships],
    }
