"""Unit tests for LangGraph agent nodes using mocked LLM responses."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.agents.state import ResearchState


def _base_state(**overrides: object) -> ResearchState:
    state: ResearchState = {
        "question": "What is retrieval-augmented generation?",
        "plan": "Research RAG definition and components.",
        "retrieved_chunks": [],
        "analysis": "",
        "fact_check_results": [],
        "draft_answer": "",
        "final_answer": "",
        "sources": [],
        "messages": [],
        "iteration": 0,
        "max_iterations": 3,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.with_structured_output.return_value = llm
    llm.ainvoke = AsyncMock()
    return llm


class TestSupervisorNode:
    async def test_produces_plan(self, mock_llm: MagicMock) -> None:
        from app.agents.supervisor import ResearchPlan, supervisor_node

        mock_llm.ainvoke.return_value = ResearchPlan(
            plan="Look up RAG definition, components, and use cases."
        )

        with patch("app.agents.supervisor.get_agent_llm", return_value=mock_llm):
            result = await supervisor_node(_base_state())

        assert "plan" in result
        assert len(result["plan"]) > 0
        assert result["iteration"] == 1

    async def test_increments_iteration(self, mock_llm: MagicMock) -> None:
        from app.agents.supervisor import ResearchPlan, supervisor_node

        mock_llm.ainvoke.return_value = ResearchPlan(plan="Short plan.")

        with patch("app.agents.supervisor.get_agent_llm", return_value=mock_llm):
            result = await supervisor_node(_base_state(iteration=2))

        assert result["iteration"] == 3


class TestResearcherNode:
    async def test_decomposes_and_retrieves(self, mock_llm: MagicMock) -> None:
        from app.agents.researcher import SubQueries, researcher_node

        mock_llm.ainvoke.return_value = SubQueries(
            queries=["What is RAG?", "How does RAG retrieval work?"]
        )

        mock_service = MagicMock()
        mock_service.retrieve = AsyncMock(
            return_value=[
                {
                    "chunk_id": "abc",
                    "content": "RAG combines retrieval with generation.",
                    "document_name": "rag_paper.pdf",
                    "chunk_index": 0,
                    "score": 0.9,
                }
            ]
        )

        state = _base_state(_retrieval_service=mock_service)  # type: ignore[typeddict-item]

        with patch("app.agents.researcher.get_agent_llm", return_value=mock_llm):
            result = await researcher_node(state)

        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) > 0
        assert mock_service.retrieve.call_count == 2

    async def test_deduplicates_chunks(self, mock_llm: MagicMock) -> None:
        from app.agents.researcher import SubQueries, researcher_node

        mock_llm.ainvoke.return_value = SubQueries(queries=["Q1", "Q2"])

        same_chunk = {
            "chunk_id": "same-id",
            "content": "Same chunk.",
            "document_name": "doc.pdf",
            "chunk_index": 0,
            "score": 0.9,
        }
        mock_service = MagicMock()
        mock_service.retrieve = AsyncMock(return_value=[same_chunk])

        state = _base_state(_retrieval_service=mock_service)  # type: ignore[typeddict-item]

        with patch("app.agents.researcher.get_agent_llm", return_value=mock_llm):
            result = await researcher_node(state)

        assert len(result["retrieved_chunks"]) == 1


class TestAnalystNode:
    async def test_produces_analysis(self, mock_llm: MagicMock) -> None:
        from app.agents.analyst import analyst_node

        mock_llm.ainvoke.return_value = AIMessage(
            content="RAG retrieves relevant documents then generates answers [Source 1]."
        )
        chunks = [
            {
                "chunk_id": "1",
                "content": "RAG = retrieval + generation.",
                "document_name": "doc.pdf",
                "chunk_index": 0,
                "score": 0.9,
            }
        ]

        with patch("app.agents.analyst.get_agent_llm", return_value=mock_llm):
            result = await analyst_node(_base_state(retrieved_chunks=chunks))

        assert "analysis" in result
        assert len(result["analysis"]) > 0

    async def test_no_chunks_returns_placeholder(self, mock_llm: MagicMock) -> None:
        from app.agents.analyst import analyst_node

        with patch("app.agents.analyst.get_agent_llm", return_value=mock_llm):
            result = await analyst_node(_base_state(retrieved_chunks=[]))

        assert "No relevant sources" in result["analysis"]
        mock_llm.ainvoke.assert_not_called()


class TestFactCheckerNode:
    async def test_returns_fact_check_results(self, mock_llm: MagicMock) -> None:
        from app.agents.fact_checker import FactCheckItem, FactCheckResults, fact_checker_node

        mock_llm.ainvoke.return_value = FactCheckResults(
            results=[
                FactCheckItem(
                    claim="RAG improves factual accuracy.",
                    supported=True,
                    confidence=0.92,
                    evidence=["Source 1 discusses accuracy gains."],
                )
            ]
        )

        chunks = [
            {
                "chunk_id": "1",
                "content": "RAG improves factual accuracy by grounding answers in retrieved docs.",
                "document_name": "doc.pdf",
                "chunk_index": 0,
                "score": 0.9,
            }
        ]

        with patch("app.agents.fact_checker.get_agent_llm", return_value=mock_llm):
            result = await fact_checker_node(
                _base_state(
                    analysis="RAG improves factual accuracy [Source 1].",
                    retrieved_chunks=chunks,
                )
            )

        assert len(result["fact_check_results"]) == 1
        assert result["fact_check_results"][0]["supported"] is True

    async def test_empty_analysis_skips_llm(self, mock_llm: MagicMock) -> None:
        from app.agents.fact_checker import fact_checker_node

        with patch("app.agents.fact_checker.get_agent_llm", return_value=mock_llm):
            result = await fact_checker_node(_base_state(analysis=""))

        assert result["fact_check_results"] == []
        mock_llm.ainvoke.assert_not_called()


class TestWriterNode:
    async def test_produces_final_answer(self, mock_llm: MagicMock) -> None:
        from app.agents.writer import writer_node

        mock_llm.ainvoke.return_value = AIMessage(
            content="RAG combines dense retrieval with generative models [Source 1]."
        )

        chunks = [
            {
                "chunk_id": "1",
                "content": "RAG = retrieval + generation.",
                "document_name": "doc.pdf",
                "chunk_index": 0,
                "score": 0.9,
            }
        ]

        with patch("app.agents.writer.get_agent_llm", return_value=mock_llm):
            result = await writer_node(
                _base_state(
                    retrieved_chunks=chunks,
                    analysis="RAG retrieves then generates [Source 1].",
                    fact_check_results=[
                        {"claim": "RAG retrieves docs", "supported": True, "confidence": 0.95, "evidence": []}
                    ],
                )
            )

        assert "final_answer" in result
        assert len(result["final_answer"]) > 0
        assert len(result["sources"]) == 1
