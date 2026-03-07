"""Module-level refs set during lifespan for MCP tool access."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.graph.store import GraphStore
    from app.ingestion.embedder import EmbeddingService
    from app.retrieval.reranker import RerankerService

graph_store: GraphStore | None = None
embedder: EmbeddingService | None = None
reranker: RerankerService | None = None
db_session_factory: async_sessionmaker[AsyncSession] | None = None
