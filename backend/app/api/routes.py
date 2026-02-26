import time
import uuid
from collections.abc import AsyncGenerator

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_embedder, get_reranker
from app.config import settings
from app.database import get_db
from app.generation.chain import generate_answer, stream_answer
from app.ingestion.embedder import EmbeddingService
from app.models.orm import Chunk, Document, EvalRun, Query
from app.models.schemas import (
    ChunkingStrategy,
    ChunkResponse,
    DocumentResponse,
    EvalMetrics,
    EvalRequest,
    EvalRunResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievalStrategy,
    SearchRequest,
    SearchResponse,
    SourceChunk,
)
from app.retrieval.reranker import RerankerService
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


async def _retrieve_chunks(
    question: str,
    strategy: RetrievalStrategy,
    top_k: int,
    db: AsyncSession,
    embedder: EmbeddingService,
    reranker: RerankerService,
    document_ids: list[uuid.UUID] | None = None,
) -> list[tuple[Chunk, float]]:
    from app.retrieval.bm25_search import bm25_search
    from app.retrieval.hybrid import hybrid_search
    from app.retrieval.vector_search import dense_search

    fetch_k = top_k * 3

    if strategy == RetrievalStrategy.DENSE:
        embedding = embedder.embed_query(question)
        results = await dense_search(embedding, db, fetch_k, document_ids)
    elif strategy == RetrievalStrategy.BM25:
        results = await bm25_search(question, db, fetch_k, document_ids)
    else:
        embedding = embedder.embed_query(question)
        results = await hybrid_search(question, embedding, db, fetch_k, settings.rrf_k, document_ids)

    if not results:
        return []

    texts = [chunk.content for chunk, _ in results]
    reranked = reranker.rerank(question, texts, top_k=top_k)
    return [(results[idx][0], score) for idx, score in reranked]


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedder),
    reranker: RerankerService = Depends(get_reranker),
) -> QueryResponse:
    start = time.monotonic()

    chunks_with_scores = await _retrieve_chunks(
        req.question, req.strategy, req.top_k, db, embedder, reranker, req.document_ids
    )

    if not chunks_with_scores:
        return QueryResponse(
            answer="No relevant documents found. Please upload documents first.",
            sources=[],
            latency_ms=0.0,
            strategy=req.strategy.value,
        )

    chunk_dicts = []
    for chunk, score in chunks_with_scores:
        doc_result = await db.execute(select(Document).where(Document.id == chunk.document_id))
        doc = doc_result.scalar_one_or_none()
        chunk_dicts.append({
            "chunk_id": chunk.id,
            "content": chunk.content,
            "document_name": doc.filename if doc else "unknown",
            "chunk_index": chunk.chunk_index,
            "score": score,
        })

    answer = await generate_answer(req.question, chunk_dicts)
    latency_ms = (time.monotonic() - start) * 1000

    query_record = Query(
        question=req.question,
        answer=answer,
        context_chunks=[c["chunk_id"] for c in chunk_dicts],
        retrieval_strategy=req.strategy.value,
        latency_ms=latency_ms,
    )
    db.add(query_record)

    sources = [
        SourceChunk(
            chunk_id=c["chunk_id"],
            content=c["content"],
            document_name=c["document_name"],
            chunk_index=c["chunk_index"],
            score=c["score"],
        )
        for c in chunk_dicts
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=latency_ms,
        strategy=req.strategy.value,
    )


@router.post("/query/stream")
async def query_stream(
    req: QueryRequest,
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedder),
    reranker: RerankerService = Depends(get_reranker),
) -> StreamingResponse:
    chunks_with_scores = await _retrieve_chunks(
        req.question, req.strategy, req.top_k, db, embedder, reranker, req.document_ids
    )

    chunk_dicts = []
    for chunk, score in chunks_with_scores:
        doc_result = await db.execute(select(Document).where(Document.id == chunk.document_id))
        doc = doc_result.scalar_one_or_none()
        chunk_dicts.append({
            "chunk_id": str(chunk.id),
            "content": chunk.content,
            "document_name": doc.filename if doc else "unknown",
            "chunk_index": chunk.chunk_index,
            "score": score,
        })

    import json

    async def event_stream() -> AsyncGenerator[str, None]:
        for cd in chunk_dicts:
            yield f"data: {json.dumps({'type': 'source', 'data': cd})}\n\n"

        if not chunk_dicts:
            yield "data: " + json.dumps({"type": "chunk", "data": "No relevant documents found."}) + "\n\n"
        else:
            async for token in stream_answer(req.question, chunk_dicts):
                yield f"data: {json.dumps({'type': 'chunk', 'data': token})}\n\n"

        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedder),
    reranker: RerankerService = Depends(get_reranker),
) -> SearchResponse:
    start = time.monotonic()
    chunks_with_scores = await _retrieve_chunks(
        req.query, req.strategy, req.top_k, db, embedder, reranker
    )
    latency_ms = (time.monotonic() - start) * 1000

    return SearchResponse(
        chunks=[
            ChunkResponse(
                id=chunk.id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                chunking_strategy=chunk.chunking_strategy,
                score=score,
            )
            for chunk, score in chunks_with_scores
        ],
        strategy=req.strategy.value,
        latency_ms=latency_ms,
    )


@router.post("/eval/run", response_model=EvalRunResponse)
async def run_eval(
    req: EvalRequest,
    db: AsyncSession = Depends(get_db),
) -> EvalRunResponse:
    import subprocess
    import sys

    result = await db.execute(select(func.count()).select_from(Document))
    doc_count = result.scalar_one()
    if doc_count == 0:
        raise HTTPException(400, "no documents ingested — upload documents before running evaluation")

    # placeholder metrics (real eval runs via CLI: python -m eval.runner)
    metrics = EvalMetrics(
        faithfulness=0.0,
        answer_relevance=0.0,
        context_precision=0.0,
        context_recall=0.0,
    )
    eval_run = EvalRun(
        dataset_name=req.dataset,
        chunking_strategy=req.chunking_strategy.value,
        retrieval_strategy=req.retrieval_strategy.value,
        metrics=metrics.model_dump(),
        sample_count=0,
    )
    db.add(eval_run)
    await db.flush()

    return EvalRunResponse(
        id=eval_run.id,
        dataset_name=eval_run.dataset_name,
        chunking_strategy=eval_run.chunking_strategy,
        retrieval_strategy=eval_run.retrieval_strategy,
        metrics=metrics,
        sample_count=0,
        created_at=eval_run.created_at,
    )


@router.get("/eval/results", response_model=list[EvalRunResponse])
async def list_eval_results(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> list[EvalRunResponse]:
    result = await db.execute(
        select(EvalRun).order_by(EvalRun.created_at.desc()).limit(limit)
    )
    runs = result.scalars().all()
    return [
        EvalRunResponse(
            id=run.id,
            dataset_name=run.dataset_name,
            chunking_strategy=run.chunking_strategy,
            retrieval_strategy=run.retrieval_strategy,
            metrics=EvalMetrics(**run.metrics),
            sample_count=run.sample_count,
            created_at=run.created_at,
        )
        for run in runs
    ]
