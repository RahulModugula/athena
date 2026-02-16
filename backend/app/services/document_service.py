import hashlib
import uuid

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunker import BaseChunker, get_chunker
from app.ingestion.embedder import EmbeddingService
from app.ingestion.loader import Page, load_document
from app.models.orm import Chunk, Document

logger = structlog.get_logger()

BATCH_SIZE = 32


async def ingest_document(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    chunking_strategy: str,
    db: AsyncSession,
    embedder: EmbeddingService,
) -> Document:
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    existing = await db.execute(
        select(Document).where(Document.content_hash == content_hash)
    )
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
        filename=filename,
        content_hash=content_hash,
        mime_type=mime_type,
        metadata_={"page_count": len(pages), "chunk_count": len(raw_chunks)},
    )
    db.add(doc)

    for batch_start in range(0, len(raw_chunks), BATCH_SIZE):
        batch = raw_chunks[batch_start:batch_start + BATCH_SIZE]
        texts = [c.content for c in batch]
        embeddings = embedder.embed_texts(texts)

        for chunk, embedding in zip(batch, embeddings):
            db_chunk = Chunk(
                document_id=doc.id,
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
    return doc
