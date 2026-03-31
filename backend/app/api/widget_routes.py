"""Public widget endpoints — designed for embeddable docs search.

These endpoints are called by the JavaScript widget embedded on customer
docs sites.  Authentication is via publishable key (pk_*) in the
X-Widget-Key header.
"""

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_embedder, get_reranker
from app.database import get_db
from app.generation.chain import generate_answer
from app.ingestion.embedder import EmbeddingService
from app.models.orm import Query, WidgetEvent
from app.retrieval.reranker import RerankerService
from app.services.retrieval_service import RetrievalService

logger = structlog.get_logger()
router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class WidgetQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    version: str | None = None


class WidgetSource(BaseModel):
    title: str
    url: str | None
    snippet: str
    chunk_id: str


class WidgetAnswer(BaseModel):
    answer: str
    sources: list[WidgetSource]
    verified: bool = False
    confidence: float = 0.0
    query_id: str


class WidgetFeedback(BaseModel):
    query_id: str
    rating: str = Field(..., pattern=r"^(up|down)$")
    comment: str = ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _require_tenant(request: Request) -> uuid.UUID:
    tid = getattr(request.state, "tenant_id", None)
    if tid is None:
        raise HTTPException(401, "valid publishable key required")
    return uuid.UUID(str(tid))


@router.post("/widget/query", response_model=WidgetAnswer)
async def widget_query(
    body: WidgetQuery,
    request: Request,
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedder),
    reranker: RerankerService = Depends(get_reranker),
) -> WidgetAnswer:
    tenant_id = _require_tenant(request)

    svc = RetrievalService(db=db, embedder=embedder, reranker=reranker, tenant_id=tenant_id)
    chunks = await svc.retrieve(body.query, top_k=5)

    if not chunks:
        query_record = Query(
            tenant_id=tenant_id,
            question=body.query,
            answer="",
            context_chunks=[],
            retrieval_strategy="hybrid",
            latency_ms=0,
        )
        db.add(query_record)
        await db.flush()
        return WidgetAnswer(
            answer=(
                "I couldn't find relevant information in the documentation. "
                "Try rephrasing your question."
            ),
            sources=[],
            query_id=str(query_record.id),
        )

    try:
        answer = await generate_answer(body.query, chunks)
    except Exception as exc:
        logger.error("widget answer generation failed", error=str(exc))
        raise HTTPException(500, "answer generation failed") from exc

    sources = [
        WidgetSource(
            title=str(c.get("document_name", "")),
            url=str(c["source_url"]) if c.get("source_url") is not None else None,
            snippet=str(c["content"])[:200],
            chunk_id=str(c["chunk_id"]),
        )
        for c in chunks
    ]

    query_record = Query(
        tenant_id=tenant_id,
        question=body.query,
        answer=answer,
        context_chunks=[c["chunk_id"] for c in chunks],
        retrieval_strategy="hybrid",
        latency_ms=0,
    )
    db.add(query_record)
    await db.flush()

    return WidgetAnswer(
        answer=answer,
        sources=sources,
        verified=False,
        confidence=0.0,
        query_id=str(query_record.id),
    )


@router.post("/widget/feedback")
async def widget_feedback(
    body: WidgetFeedback,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict[str, bool]:
    tenant_id = getattr(request.state, "tenant_id", None)
    event = WidgetEvent(
        tenant_id=tenant_id,
        query_id=uuid.UUID(body.query_id) if body.query_id else None,
        event_type=f"feedback_{body.rating}",
        payload={"comment": body.comment} if body.comment else {},
    )
    db.add(event)
    return {"ok": True}
