"""Tests for MCP tool implementations."""

from unittest.mock import AsyncMock, MagicMock, patch


class TestAthenaSearchDocuments:
    async def test_returns_empty_when_no_refs(self) -> None:
        with patch("app.mcp._store_ref.db_session_factory", None):
            from app.mcp.server import athena_search_documents
            result = await athena_search_documents("test query")
        assert result == []

    async def test_returns_results_when_service_available(self) -> None:
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_factory = MagicMock()
        mock_factory.return_value = mock_session

        mock_embedder = MagicMock()
        mock_reranker = MagicMock()

        fake_chunks = [
            {
                "chunk_id": "abc-123",
                "content": "RAG combines retrieval with generation.",
                "document_name": "rag_paper.pdf",
                "chunk_index": 0,
                "score": 0.92,
            }
        ]

        with (
            patch("app.mcp._store_ref.db_session_factory", mock_factory),
            patch("app.mcp._store_ref.embedder", mock_embedder),
            patch("app.mcp._store_ref.reranker", mock_reranker),
            patch("app.services.retrieval_service.RetrievalService") as mock_svc_cls,
        ):
            instance = AsyncMock()
            instance.retrieve = AsyncMock(return_value=fake_chunks)
            mock_svc_cls.return_value = instance

            from app.mcp.server import athena_search_documents
            result = await athena_search_documents("what is RAG?", top_k=3)

        assert len(result) == 1
        assert result[0]["document_name"] == "rag_paper.pdf"
        assert result[0]["score"] == 0.92
        assert isinstance(result[0]["chunk_id"], str)

    async def test_serialises_chunk_id_to_string(self) -> None:
        """chunk_id UUIDs must be JSON-serialisable strings."""
        import uuid

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_factory = MagicMock(return_value=mock_session)

        uid = uuid.uuid4()
        fake_chunks = [
            {
                "chunk_id": uid,
                "content": "some content",
                "document_name": "doc.pdf",
                "chunk_index": 0,
                "score": 0.5,
            }
        ]

        with (
            patch("app.mcp._store_ref.db_session_factory", mock_factory),
            patch("app.mcp._store_ref.embedder", MagicMock()),
            patch("app.mcp._store_ref.reranker", MagicMock()),
            patch("app.services.retrieval_service.RetrievalService") as mock_svc_cls,
        ):
            instance = AsyncMock()
            instance.retrieve = AsyncMock(return_value=fake_chunks)
            mock_svc_cls.return_value = instance

            from app.mcp.server import athena_search_documents
            result = await athena_search_documents("query")

        assert isinstance(result[0]["chunk_id"], str)
        assert result[0]["chunk_id"] == str(uid)


class TestAthenaQueryKnowledgeGraph:
    async def test_returns_error_when_no_graph_store(self) -> None:
        with patch("app.mcp._store_ref.graph_store", None):
            from app.mcp.server import athena_query_knowledge_graph
            result = await athena_query_knowledge_graph("Python")
        assert "error" in result

    async def test_returns_subgraph_when_connected(self) -> None:
        mock_store = AsyncMock()
        mock_store.is_connected = True

        mock_entity = MagicMock()
        mock_entity.model_dump.return_value = {"name": "Python", "type": "Language"}
        mock_rel = MagicMock()
        mock_rel.model_dump.return_value = {"source": "Python", "relation": "USED_IN", "target": "RAG"}

        mock_subgraph = MagicMock()
        mock_subgraph.entities = [mock_entity]
        mock_subgraph.relationships = [mock_rel]
        mock_store.get_entity_context = AsyncMock(return_value=mock_subgraph)

        with patch("app.mcp._store_ref.graph_store", mock_store):
            from app.mcp.server import athena_query_knowledge_graph
            result = await athena_query_knowledge_graph("Python")

        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Python"
        assert len(result["relationships"]) == 1
