import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import settings
from app.models.orm import Document
from app.models.schemas import HealthResponse

logger = structlog.get_logger()

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    result = await db.execute(select(func.count()).select_from(Document))
    doc_count = result.scalar_one()
    return HealthResponse(
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        document_count=doc_count,
    )
