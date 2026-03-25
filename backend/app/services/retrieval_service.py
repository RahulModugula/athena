import uuid

import structlog
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.ingestion.embedder import EmbeddingService
from app.models.orm import Chunk, Document
from app.models.schemas import RetrievalStrategy
from app.retrieval.reranker import RerankerService

logger = structlog.get_logger()

_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a technical documentation expert. Generate a concise, factual passage "
        "that would directly answer the following question. Write as if it were an excerpt "
        "from a technical document. Do not include phrases like 'This document explains' "
        "or 'The answer is'. Just write the passage itself.",
    ),
    ("human", "{question}"),
])


class RetrievalService:
    def __init__(
        self,
        db: AsyncSession,
        embedder: EmbeddingService,
        reranker: RerankerService,
        tenant_id: uuid.UUID | None = None,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.reranker = reranker
        self.tenant_id = tenant_id

    async def retrieve(
        self,
        question: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        top_k: int | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[dict[str, object]]:
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
                "source_url": doc.source_url if doc else None,
                "chunk_index": chunk.chunk_index,
                "score": score,
            })
        return result

    async def _hyde_embed(self, question: str) -> list[float]:
        """Generate a hypothetical answer document and embed it instead of the raw query.

        HyDE (Gao et al., arXiv 2212.10496) closes the gap between short sparse queries
        and dense document representations. Rather than embedding the query string directly,
        an LLM generates a plausible answer passage; that passage is embedded and used for
        nearest-neighbour search. The hypothetical document lives in the same vector space
        as real documents, so cosine similarity is far more meaningful than query→doc similarity.
        """
        from app.generation.chain import get_llm

        llm = get_llm(streaming=False)
        chain = _HYDE_PROMPT | llm | StrOutputParser()
        hypothetical_doc: str = await chain.ainvoke({"question": question})
        logger.debug("hyde generated hypothetical doc", length=len(hypothetical_doc))
        return self.embedder.embed_query(hypothetical_doc)

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
            results = await dense_search(
                embedding, self.db, fetch_k, document_ids, self.tenant_id
            )
        elif strategy == RetrievalStrategy.BM25:
            results = await bm25_search(
                question, self.db, fetch_k, document_ids, self.tenant_id
            )
        elif strategy == RetrievalStrategy.HYDE:
            # Embed a hypothetical answer document; rerank with BM25 for final results
            hyde_embedding = await self._hyde_embed(question)
            dense_results = await dense_search(
                hyde_embedding, self.db, fetch_k, document_ids, self.tenant_id
            )
            bm25_results = await bm25_search(
                question, self.db, fetch_k, document_ids, self.tenant_id
            )
            # Fuse HyDE dense results with BM25 for complementary coverage
            from app.retrieval.hybrid import reciprocal_rank_fusion
            dense_ranked = [(c.id, s) for c, s in dense_results]
            bm25_ranked = [(c.id, s) for c, s in bm25_results]
            fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=settings.rrf_k)
            chunk_map = {c.id: c for c, _ in dense_results}
            chunk_map.update({c.id: c for c, _ in bm25_results})
            results = [
                (chunk_map[cid], score)
                for cid, score in fused[:fetch_k]
                if cid in chunk_map
            ]
        else:
            # Default: HYBRID (dense + BM25 with RRF)
            embedding = self.embedder.embed_query(question)
            results = await hybrid_search(
                question, embedding, self.db, fetch_k, settings.rrf_k,
                document_ids, self.tenant_id,
            )

        if not results:
            return []

        texts = [chunk.content for chunk, _ in results]
        reranked = self.reranker.rerank(question, texts, top_k=top_k)
        return [(results[idx][0], score) for idx, score in reranked]
