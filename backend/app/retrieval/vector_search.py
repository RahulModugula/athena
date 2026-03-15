import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import Chunk

logger = structlog.get_logger()


async def dense_search(
    query_embedding: list[float],
    db: AsyncSession,
    top_k: int = 10,
    document_ids: list | None = None,
) -> list[tuple[Chunk, float]]:
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    query = (
        select(
            Chunk,
            (1 - Chunk.embedding.cosine_distance(embedding_str)).label("score"),
        )
        .order_by(Chunk.embedding.cosine_distance(embedding_str))
        .limit(top_k)
    )

    if document_ids:
        query = query.where(Chunk.document_id.in_(document_ids))

    result = await db.execute(query)
    rows = result.all()

    return [(row[0], float(row[1])) for row in rows]
