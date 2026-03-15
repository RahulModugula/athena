from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph

from app.agents.analyst import analyst_node
from app.agents.fact_checker import fact_checker_node
from app.agents.researcher import researcher_node
from app.agents.state import ResearchState
from app.agents.supervisor import supervisor_node
from app.agents.writer import writer_node


def build_research_graph() -> StateGraph[Any]:
    graph: StateGraph[Any] = StateGraph(ResearchState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "fact_checker")
    graph.add_edge("fact_checker", "writer")
    graph.add_edge("writer", END)

    return graph


_compiled_graph: CompiledGraph | None = None


def get_research_graph() -> CompiledGraph:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_research_graph().compile()
    return _compiled_graph
