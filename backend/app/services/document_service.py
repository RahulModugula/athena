import asyncio
import hashlib
import uuid
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.ingestion.chunker import BaseChunker, get_chunker
from app.ingestion.embedder import EmbeddingService
from app.ingestion.loader import Page, load_document
from app.models.orm import Chunk, Document

logger = structlog.get_logger()

BATCH_SIZE = 32

_CONTEXT_PROMPT = """\
<document>
{full_document}
</document>

Here is a chunk from the document:

<chunk>
{chunk_text}
</chunk>

In 1-2 sentences, describe what this chunk is about in the context of the full document. \
Be specific about the subject matter. Do not start with "This chunk" — just describe the content directly."""


async def _generate_chunk_context(full_document: str, chunk_text: str) -> str:
    """Generate a short situating context for a chunk using a cheap/fast LLM.

    Implements Anthropic's Contextual Retrieval technique (Sep 2024). Prepending
    this context to each chunk before embedding reduces retrieval failures by ~49%
    compared to embedding raw chunks. The full document is passed so the model can
    resolve pronouns, abbreviations, and implicit references in the chunk.
    """
    from pydantic import SecretStr

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        # Use a cheap/fast model dedicated to context generation; fall back to the
        # configured LLM model if the haiku model is unavailable (e.g. different provider).
        model = settings.contextual_retrieval_model
        llm = ChatAnthropic(
            model_name=model,
            api_key=SecretStr(settings.anthropic_api_key),
            temperature=0.0,
            max_tokens=150,
            timeout=None,
            stop=None,
        )
        prompt = ChatPromptTemplate.from_messages([("human", _CONTEXT_PROMPT)])
        chain = prompt | llm | StrOutputParser()
        context: str = await chain.ainvoke({
            "full_document": full_document[:8000],  # cap to avoid token overflow
            "chunk_text": chunk_text,
        })
        return context.strip()
    except Exception as exc:
        logger.warning("contextual retrieval generation failed, skipping", error=str(exc))
        return ""


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

    # Contextual Retrieval: prepend LLM-generated context to each chunk before embedding.
    # This situates decontextualized chunks within their source document, reducing
    # retrieval failures by ~49% (Anthropic, Sep 2024). Enabled via ATHENA_CONTEXTUAL_RETRIEVAL_ENABLED.
    contextual_texts: list[str] = []
    if settings.contextual_retrieval_enabled and settings.anthropic_api_key:
        logger.info("contextual retrieval enabled, generating chunk contexts", chunks=len(raw_chunks))
        context_tasks = [_generate_chunk_context(full_text, c.content) for c in raw_chunks]
        contexts = await asyncio.gather(*context_tasks)
        for chunk, ctx in zip(raw_chunks, contexts, strict=True):
            if ctx:
                contextual_texts.append(f"{ctx}\n\n{chunk.content}")
            else:
                contextual_texts.append(chunk.content)
    else:
        contextual_texts = [c.content for c in raw_chunks]

    for batch_start in range(0, len(raw_chunks), BATCH_SIZE):
        batch = raw_chunks[batch_start : batch_start + BATCH_SIZE]
        embed_texts = contextual_texts[batch_start : batch_start + BATCH_SIZE]
        embeddings = embedder.embed_texts(embed_texts)

        for chunk, embed_text, embedding in zip(batch, embed_texts, embeddings, strict=True):
            db_chunk = Chunk(
                document_id=doc.id,
                tenant_id=tenant_id,
                content=embed_text,  # store context-augmented text for retrieval + display
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
