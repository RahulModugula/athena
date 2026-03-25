import uuid

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import Chunk

logger = structlog.get_logger()


async def bm25_search(
    query_text: str,
    db: AsyncSession,
    top_k: int = 10,
    document_ids: list | None = None,
    tenant_id: uuid.UUID | None = None,
) -> list[tuple[Chunk, float]]:
    tsquery = func.plainto_tsquery("english", query_text)
    tsvector = func.to_tsvector("english", Chunk.content)
    rank = func.ts_rank_cd(tsvector, tsquery)

    query = (
        select(Chunk, rank.label("score"))
        .where(tsvector.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(top_k)
    )

    if tenant_id is not None:
        query = query.where(Chunk.tenant_id == tenant_id)

    if document_ids:
        query = query.where(Chunk.document_id.in_(document_ids))

    result = await db.execute(query)
    rows = result.all()

    return [(row[0], float(row[1])) for row in rows]
