import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.ingestion.embedder import EmbeddingService
from app.models.orm import Chunk, Document
from app.models.schemas import RetrievalStrategy
from app.retrieval.reranker import RerankerService


class RetrievalService:
    def __init__(
        self,
        db: AsyncSession,
        embedder: EmbeddingService,
        reranker: RerankerService,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.reranker = reranker

    async def retrieve(
        self,
        question: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        top_k: int | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[dict]:
        top_k = top_k or settings.rerank_top_k
        chunks_with_scores = await self._retrieve_chunks(
            question, strategy, top_k, document_ids
        )
        result = []
        for chunk, score in chunks_with_scores:
            doc_result = await self.db.execute(
                select(Document).where(Document.id == chunk.document_id)
            )
            doc = doc_result.scalar_one_or_none()
            result.append({
                "chunk_id": chunk.id,
                "content": chunk.content,
                "document_name": doc.filename if doc else "unknown",
                "chunk_index": chunk.chunk_index,
                "score": score,
            })
        return result

    async def _retrieve_chunks(
        self,
        question: str,
        strategy: RetrievalStrategy,
        top_k: int,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[tuple[Chunk, float]]:
        from app.retrieval.bm25_search import bm25_search
        from app.retrieval.hybrid import hybrid_search
        from app.retrieval.vector_search import dense_search

        fetch_k = top_k * 3

        if strategy == RetrievalStrategy.DENSE:
            embedding = self.embedder.embed_query(question)
            results = await dense_search(embedding, self.db, fetch_k, document_ids)
        elif strategy == RetrievalStrategy.BM25:
            results = await bm25_search(question, self.db, fetch_k, document_ids)
        else:
            embedding = self.embedder.embed_query(question)
            results = await hybrid_search(
                question, embedding, self.db, fetch_k, settings.rrf_k, document_ids
            )

        if not results:
            return []

        texts = [chunk.content for chunk, _ in results]
        reranked = self.reranker.rerank(question, texts, top_k=top_k)
        return [(results[idx][0], score) for idx, score in reranked]
