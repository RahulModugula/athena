"""Integration tests for the FastAPI routes using mocked dependencies.

These tests validate route logic, HTTP semantics, and response shapes without
requiring a running database or ML models.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_embedder, get_reranker
from app.database import get_db  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedder() -> MagicMock:
    emb = MagicMock()
    emb.embed_query.return_value = [0.0] * 1024
    return emb


@pytest.fixture
def mock_reranker() -> MagicMock:
    rr = MagicMock()
    rr.rerank.return_value = []
    return rr


@pytest.fixture
def mock_db() -> AsyncMock:
    db = AsyncMock()
    # Use a plain MagicMock for the result so .scalar_one(), .scalars().all()
    # etc. are synchronous (matching the real SQLAlchemy Result interface).
    result = MagicMock()
    result.scalar_one.return_value = 0
    result.scalar_one_or_none.return_value = None
    result.scalars.return_value.all.return_value = []
    result.all.return_value = []
    db.execute = AsyncMock(return_value=result)
    return db


@pytest.fixture
async def client(mock_db: AsyncMock, mock_embedder: MagicMock, mock_reranker: MagicMock) -> AsyncClient:
    from app.main import app

    async def _override_db():
        yield mock_db

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_reranker] = lambda: mock_reranker

    # Prevent lifespan from loading heavy ML models
    with (
        patch("app.ingestion.embedder.EmbeddingService"),
        patch("app.retrieval.reranker.RerankerService"),
        patch("app.observability.logging.configure_logging"),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_returns_healthy(self, client: AsyncClient, mock_db: AsyncMock) -> None:
        mock_db.execute.return_value.scalar_one.return_value = 3
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "embedding_model" in data
        assert "document_count" in data


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------


class TestDocumentUpload:
    async def test_unsupported_mime_returns_400(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/documents/upload",
            files={"file": ("test.exe", b"MZ", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"]

    async def test_empty_file_returns_400(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert response.status_code == 400

    async def test_duplicate_returns_409(self, client: AsyncClient) -> None:
        with patch("app.api.routes.ingest_document") as mock_ingest:
            mock_ingest.side_effect = ValueError("document already exists")
            response = await client.post(
                "/api/documents/upload",
                files={"file": ("doc.txt", b"hello world", "text/plain")},
            )
        assert response.status_code == 409

    async def test_valid_upload_returns_document(self, client: AsyncClient, mock_db: AsyncMock) -> None:
        doc_id = uuid.uuid4()
        mock_doc = MagicMock()
        mock_doc.id = doc_id
        mock_doc.filename = "test.txt"
        mock_doc.mime_type = "text/plain"
        mock_doc.metadata_ = {}
        mock_doc.created_at = datetime.now(UTC)

        mock_db.execute.return_value.scalar_one.return_value = 5  # chunk count

        with patch("app.api.routes.ingest_document", AsyncMock(return_value=mock_doc)):
            response = await client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", b"some document content", "text/plain")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(doc_id)
        assert data["filename"] == "test.txt"
        assert data["chunk_count"] == 5


class TestListDocuments:
    async def test_returns_empty_list(self, client: AsyncClient) -> None:
        response = await client.get("/api/documents")
        assert response.status_code == 200
        assert response.json() == []

    async def test_returns_document_list(self, client: AsyncClient, mock_db: AsyncMock) -> None:
        doc_id = uuid.uuid4()
        mock_doc = MagicMock()
        mock_doc.id = doc_id
        mock_doc.filename = "paper.pdf"
        mock_doc.mime_type = "application/pdf"
        mock_doc.metadata_ = {}
        mock_doc.created_at = datetime.now(UTC)
        mock_doc.chunks = []
        mock_db.execute.return_value.scalars.return_value.all.return_value = [mock_doc]

        response = await client.get("/api/documents")
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["filename"] == "paper.pdf"


class TestDeleteDocument:
    async def test_not_found_returns_404(self, client: AsyncClient) -> None:
        doc_id = uuid.uuid4()
        response = await client.delete(f"/api/documents/{doc_id}")
        assert response.status_code == 404

    async def test_existing_document_deleted(self, client: AsyncClient, mock_db: AsyncMock) -> None:
        doc_id = uuid.uuid4()
        mock_doc = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = mock_doc

        response = await client.delete(f"/api/documents/{doc_id}")
        assert response.status_code == 200
        assert response.json() == {"deleted": True}
        mock_db.delete.assert_called_once_with(mock_doc)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestQuery:
    async def test_no_documents_returns_placeholder(self, client: AsyncClient) -> None:
        with patch("app.api.routes._retrieve_chunks", AsyncMock(return_value=[])):
            response = await client.post(
                "/api/query",
                json={"question": "What is RAG?"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "No relevant documents" in data["answer"]
        assert data["sources"] == []

    async def test_returns_answer_with_sources(
        self, client: AsyncClient, mock_db: AsyncMock
    ) -> None:
        chunk_id = uuid.uuid4()
        mock_chunk = MagicMock()
        mock_chunk.id = chunk_id
        mock_chunk.content = "RAG retrieves relevant documents."
        mock_chunk.chunk_index = 0
        mock_chunk.document_id = uuid.uuid4()

        mock_doc = MagicMock()
        mock_doc.id = mock_chunk.document_id
        mock_doc.filename = "rag.pdf"
        # The route batch-loads documents via scalars().all()
        mock_db.execute.return_value.scalars.return_value.all.return_value = [mock_doc]

        with (
            patch("app.api.routes._retrieve_chunks", AsyncMock(return_value=[(mock_chunk, 0.9)])),
            patch("app.api.routes.generate_answer", AsyncMock(return_value="RAG is great.")),
        ):
            response = await client.post(
                "/api/query",
                json={"question": "What is RAG?", "strategy": "hybrid"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "RAG is great."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["document_name"] == "rag.pdf"
        assert data["strategy"] == "hybrid"

    async def test_top_k_validation(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/query",
            json={"question": "test", "top_k": 0},
        )
        assert response.status_code == 422  # pydantic validation

    async def test_invalid_strategy_returns_422(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/query",
            json={"question": "test", "strategy": "foobar"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    async def test_returns_chunks(self, client: AsyncClient) -> None:
        chunk_id = uuid.uuid4()
        mock_chunk = MagicMock()
        mock_chunk.id = chunk_id
        mock_chunk.content = "Some chunk content"
        mock_chunk.chunk_index = 2
        mock_chunk.token_count = 80
        mock_chunk.chunking_strategy = "recursive"

        with patch("app.api.routes._retrieve_chunks", AsyncMock(return_value=[(mock_chunk, 0.85)])):
            response = await client.post(
                "/api/search",
                json={"query": "machine learning", "top_k": 5},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["score"] == pytest.approx(0.85)
        assert "latency_ms" in data


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestEval:
    async def test_run_requires_documents(self, client: AsyncClient, mock_db: AsyncMock) -> None:
        mock_db.execute.return_value.scalar_one.return_value = 0
        response = await client.post(
            "/api/eval/run",
            json={"dataset": "sample_qa"},
        )
        assert response.status_code == 400
        assert "no documents" in response.json()["detail"]

    async def test_results_returns_list(self, client: AsyncClient, mock_db: AsyncMock) -> None:
        mock_db.execute.return_value.scalars.return_value.all.return_value = []
        response = await client.get("/api/eval/results")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
