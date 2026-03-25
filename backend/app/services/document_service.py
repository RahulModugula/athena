import asyncio
import hashlib
import uuid
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunker import BaseChunker, get_chunker
from app.ingestion.embedder import EmbeddingService
from app.ingestion.loader import Page, load_document
from app.models.orm import Chunk, Document

logger = structlog.get_logger()

BATCH_SIZE = 32


async def _extract_and_store(chunks: list[Any], graph_store: Any, llm: Any) -> None:
    """Fire-and-forget task to extract entities from chunks and store in graph."""
    from app.graph.extractor import extract_entities

    for chunk in chunks:
        try:
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            entities, relationships = await extract_entities(text, llm)
            if entities:
                await graph_store.upsert_entities(entities)
            if relationships:
                await graph_store.upsert_relationships(relationships)
        except Exception as exc:
            logger.warning("graph extraction failed for chunk", error=str(exc))


async def ingest_document(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    chunking_strategy: str,
    db: AsyncSession,
    embedder: EmbeddingService,
    tenant_id: uuid.UUID | None = None,
    graph_store: Any = None,
    llm: Any = None,
    source_url: str | None = None,
    doc_version: str | None = None,
) -> Document:
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    dup_query = select(Document).where(Document.content_hash == content_hash)
    if tenant_id is not None:
        dup_query = dup_query.where(Document.tenant_id == tenant_id)
    existing = await db.execute(dup_query)
    if existing.scalar_one_or_none():
        raise ValueError(f"document already ingested: {filename}")

    pages: list[Page] = load_document(file_bytes, filename, mime_type)
    full_text = "\n\n".join(p.content for p in pages)

    if chunking_strategy == "semantic":
        chunker: BaseChunker = get_chunker(chunking_strategy, embedding_service=embedder)
    else:
        chunker = get_chunker(chunking_strategy)

    raw_chunks = chunker.chunk(full_text, metadata={"source": filename})

    doc = Document(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        filename=filename,
        content_hash=content_hash,
        mime_type=mime_type,
        source_url=source_url,
        doc_version=doc_version,
        metadata_={"page_count": len(pages), "chunk_count": len(raw_chunks)},
    )
    db.add(doc)

    for batch_start in range(0, len(raw_chunks), BATCH_SIZE):
        batch = raw_chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c.content for c in batch]
        embeddings = embedder.embed_texts(texts)

        for chunk, embedding in zip(batch, embeddings, strict=True):
            db_chunk = Chunk(
                document_id=doc.id,
                tenant_id=tenant_id,
                content=chunk.content,
                chunk_index=chunk.index,
                token_count=chunk.token_count,
                chunking_strategy=chunking_strategy,
                metadata_=chunk.metadata,
                embedding=embedding,
            )
            db.add(db_chunk)

    await db.flush()
    logger.info(
        "document ingested",
        filename=filename,
        chunks=len(raw_chunks),
        strategy=chunking_strategy,
    )

    if graph_store is not None and llm is not None:
        asyncio.create_task(_extract_and_store(raw_chunks[:3], graph_store, llm))

    return doc
