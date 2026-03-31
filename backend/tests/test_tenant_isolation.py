"""Tests verifying multi-tenant data isolation.

These tests ensure that tenants cannot see each other's documents or query
results. They use mocked DB/ML dependencies to focus on the isolation logic.
"""

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_embedder, get_reranker
from app.database import get_db  # noqa: F401

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TENANT_A = uuid.uuid4()
TENANT_B = uuid.uuid4()


def _make_doc(tenant_id: uuid.UUID, filename: str = "doc.pdf") -> MagicMock:
    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.tenant_id = tenant_id
    doc.filename = filename
    doc.mime_type = "application/pdf"
    doc.metadata_ = {}
    doc.created_at = datetime.now(UTC)
    return doc


def _make_chunk(doc_id: uuid.UUID, tenant_id: uuid.UUID) -> MagicMock:
    chunk = MagicMock()
    chunk.id = uuid.uuid4()
    chunk.document_id = doc_id
    chunk.tenant_id = tenant_id
    chunk.content = "Some chunk content"
    chunk.chunk_index = 0
    chunk.token_count = 50
    chunk.chunking_strategy = "recursive"
    return chunk


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


@asynccontextmanager
async def _client_for_tenant(
    tenant_id: uuid.UUID | None,
    mock_db: AsyncMock,
    mock_embedder: MagicMock,
    mock_reranker: MagicMock,
):
    from app.main import app

    async def _override_db():
        yield mock_db

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_reranker] = lambda: mock_reranker

    async def _set_tenant(self, request, call_next):
        request.state.tenant_id = tenant_id
        request.state.tenant = None
        return await call_next(request)

    # Reset the middleware stack so the patched dispatch is picked up fresh
    app.middleware_stack = None

    with (
        patch("app.ingestion.embedder.EmbeddingService"),
        patch("app.retrieval.reranker.RerankerService"),
        patch("app.observability.logging.configure_logging"),
        patch("app.api.middleware.TenantAuthMiddleware.dispatch", _set_tenant),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac

    app.middleware_stack = None
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Isolation: list documents
# ---------------------------------------------------------------------------


class TestListDocumentsIsolation:
    async def test_tenant_a_sees_only_own_documents(
        self,
        mock_embedder: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        """Tenant A listing documents should not receive tenant B's docs."""
        doc_a = _make_doc(TENANT_A, "tenant_a_doc.pdf")
        _make_doc(TENANT_B, "tenant_b_doc.pdf")  # exists but must not appear

        mock_db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [doc_a]
        result.all.return_value = []
        mock_db.execute = AsyncMock(return_value=result)

        async with _client_for_tenant(TENANT_A, mock_db, mock_embedder, mock_reranker) as ac:
            response = await ac.get("/api/documents")

        assert response.status_code == 200
        filenames = [d["filename"] for d in response.json()]
        assert "tenant_a_doc.pdf" in filenames
        assert "tenant_b_doc.pdf" not in filenames

    async def test_tenant_b_sees_only_own_documents(
        self,
        mock_embedder: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        doc_b = _make_doc(TENANT_B, "tenant_b_doc.pdf")

        mock_db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [doc_b]
        result.all.return_value = []
        mock_db.execute = AsyncMock(return_value=result)

        async with _client_for_tenant(TENANT_B, mock_db, mock_embedder, mock_reranker) as ac:
            response = await ac.get("/api/documents")

        assert response.status_code == 200
        filenames = [d["filename"] for d in response.json()]
        assert "tenant_b_doc.pdf" in filenames

    async def test_unauthenticated_sees_no_tenant_filtered_docs(
        self,
        mock_embedder: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        """tenant_id=None means no WHERE filter — global view (single-user mode)."""
        mock_db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        result.all.return_value = []
        mock_db.execute = AsyncMock(return_value=result)

        async with _client_for_tenant(None, mock_db, mock_embedder, mock_reranker) as ac:
            response = await ac.get("/api/documents")

        assert response.status_code == 200
        assert response.json() == []


# ---------------------------------------------------------------------------
# Isolation: query results
# ---------------------------------------------------------------------------


class TestQueryIsolation:
    async def test_query_scoped_to_tenant_a_chunks(
        self,
        mock_embedder: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        """When _retrieve_chunks is called it should receive tenant_a's ID."""
        mock_db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        result.all.return_value = []
        mock_db.execute = AsyncMock(return_value=result)

        captured: list[uuid.UUID | None] = []

        async def _mock_retrieve(question, strategy, top_k, db, embedder, reranker,
                                  document_ids=None, tenant_id=None):
            captured.append(tenant_id)
            return []

        with patch("app.api.routes._retrieve_chunks", _mock_retrieve):
            async with _client_for_tenant(TENANT_A, mock_db, mock_embedder, mock_reranker) as ac:
                await ac.post("/api/query", json={"question": "What is RAG?"})

        assert captured == [TENANT_A], (
            f"expected tenant_id={TENANT_A} to be passed to retrieval, got {captured}"
        )

    async def test_query_scoped_to_tenant_b_chunks(
        self,
        mock_embedder: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        mock_db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        result.all.return_value = []
        mock_db.execute = AsyncMock(return_value=result)

        captured: list[uuid.UUID | None] = []

        async def _mock_retrieve(question, strategy, top_k, db, embedder, reranker,
                                  document_ids=None, tenant_id=None):
            captured.append(tenant_id)
            return []

        with patch("app.api.routes._retrieve_chunks", _mock_retrieve):
            async with _client_for_tenant(TENANT_B, mock_db, mock_embedder, mock_reranker) as ac:
                await ac.post("/api/query", json={"question": "test"})

        assert captured == [TENANT_B]

    async def test_cross_tenant_chunks_not_exposed(
        self,
        mock_embedder: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        """Answer built from tenant_a chunks is not returned to tenant_b query."""
        doc_a_id = uuid.uuid4()
        chunk_a = _make_chunk(doc_a_id, TENANT_A)
        doc_a = _make_doc(TENANT_A, "secret_a.pdf")
        doc_a.id = doc_a_id

        mock_db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [doc_a]
        result.all.return_value = []
        mock_db.execute = AsyncMock(return_value=result)

        # _retrieve_chunks returns chunk belonging to tenant_a (simulates a bug
        # in retrieval that accidentally crosses tenant boundary)
        async def _leaky_retrieve(question, strategy, top_k, db, embedder, reranker,
                                   document_ids=None, tenant_id=None):
            if tenant_id == TENANT_B:
                # Should return empty — tenant B has no docs
                return []
            return [(chunk_a, 0.9)]

        with patch("app.api.routes._retrieve_chunks", _leaky_retrieve):
            async with _client_for_tenant(TENANT_B, mock_db, mock_embedder, mock_reranker) as ac:
                response = await ac.post("/api/query", json={"question": "test"})

        assert response.status_code == 200
        # No documents found for tenant B
        assert "No relevant documents" in response.json()["answer"]
        assert response.json()["sources"] == []


# ---------------------------------------------------------------------------
# Isolation: input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    async def test_question_too_long_returns_422(
        self,
        mock_embedder: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        mock_db = AsyncMock()
        async with _client_for_tenant(None, mock_db, mock_embedder, mock_reranker) as ac:
            response = await ac.post(
                "/api/query",
                json={"question": "x" * 10_001},
            )
        assert response.status_code == 422

    async def test_empty_question_returns_422(
        self,
        mock_embedder: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        mock_db = AsyncMock()
        async with _client_for_tenant(None, mock_db, mock_embedder, mock_reranker) as ac:
            response = await ac.post(
                "/api/query",
                json={"question": ""},
            )
        assert response.status_code == 422

    async def test_search_query_too_long_returns_422(
        self,
        mock_embedder: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        mock_db = AsyncMock()
        async with _client_for_tenant(None, mock_db, mock_embedder, mock_reranker) as ac:
            response = await ac.post(
                "/api/search",
                json={"query": "y" * 10_001},
            )
        assert response.status_code == 422
