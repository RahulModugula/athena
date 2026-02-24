import uuid

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import Chunk
from app.retrieval.bm25_search import bm25_search
from app.retrieval.vector_search import dense_search

logger = structlog.get_logger()


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[uuid.UUID, float]]],
    k: int = 60,
) -> list[tuple[uuid.UUID, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum(1 / (k + rank_i(d))) for each list i.
    k=60 is the empirically optimal constant from the original paper.
    """
    scores: dict[uuid.UUID, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _original_score) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


async def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    db: AsyncSession,
    top_k: int = 10,
    rrf_k: int = 60,
    document_ids: list | None = None,
) -> list[tuple[Chunk, float]]:
    """Run dense + BM25 search, merge results with Reciprocal Rank Fusion."""
    fetch_k = top_k * 5  # over-fetch for better fusion

    dense_results = await dense_search(query_embedding, db, fetch_k, document_ids)
    bm25_results = await bm25_search(query_text, db, fetch_k, document_ids)

    dense_ranked = [(chunk.id, score) for chunk, score in dense_results]
    bm25_ranked = [(chunk.id, score) for chunk, score in bm25_results]

    fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=rrf_k)

    chunk_map: dict[uuid.UUID, Chunk] = {}
    for chunk, _score in dense_results:
        chunk_map[chunk.id] = chunk
    for chunk, _score in bm25_results:
        chunk_map[chunk.id] = chunk

    results = []
    for chunk_id, rrf_score in fused[:top_k]:
        if chunk_id in chunk_map:
            results.append((chunk_map[chunk_id], rrf_score))

    logger.info(
        "hybrid search complete",
        dense_count=len(dense_results),
        bm25_count=len(bm25_results),
        fused_count=len(results),
    )
    return results
