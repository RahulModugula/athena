import uuid
import time

import structlog
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.config import settings
from app.api.deps import get_embedder
from app.ingestion.embedder import EmbeddingService
from app.models.orm import Chunk, Document
from app.models.schemas import (
    ChunkingStrategy,
    DocumentResponse,
    HealthResponse,
)
from app.services.document_service import ingest_document

logger = structlog.get_logger()

router = APIRouter()

ALLOWED_MIME_TYPES = {"application/pdf", "text/plain", "text/markdown"}


@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    result = await db.execute(select(func.count()).select_from(Document))
    doc_count = result.scalar_one()
    return HealthResponse(
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        document_count=doc_count,
    )


@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: ChunkingStrategy = Form(ChunkingStrategy.RECURSIVE),
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedder),
) -> DocumentResponse:
    mime_type = file.content_type or "text/plain"
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(400, f"unsupported file type: {mime_type}")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(400, "empty file")
    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(400, "file too large (max 50MB)")

    try:
        doc = await ingest_document(
            file_bytes=file_bytes,
            filename=file.filename or "untitled",
            mime_type=mime_type,
            chunking_strategy=chunking_strategy.value,
            db=db,
            embedder=embedder,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    chunk_count_result = await db.execute(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id)
    )
    chunk_count = chunk_count_result.scalar_one()

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        mime_type=doc.mime_type,
        metadata=doc.metadata_,
        chunk_count=chunk_count,
        created_at=doc.created_at,
    )


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
) -> list[DocumentResponse]:
    result = await db.execute(
        select(Document)
        .options(selectinload(Document.chunks))
        .offset(skip)
        .limit(limit)
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return [
        DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            mime_type=doc.mime_type,
            metadata=doc.metadata_,
            chunk_count=len(doc.chunks),
            created_at=doc.created_at,
        )
        for doc in docs
    ]


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, bool]:
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(404, "document not found")
    await db.delete(doc)
    return {"deleted": True}
