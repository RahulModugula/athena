from abc import ABC, abstractmethod
from dataclasses import dataclass

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = structlog.get_logger()


@dataclass
class Chunk:
    content: str
    index: int
    token_count: int
    metadata: dict


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        ...


class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 100) -> None:
        self.chunk_size = chunk_size * 4  # approximate chars from tokens
        self.overlap = overlap * 4

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    content=chunk_text,
                    index=idx,
                    token_count=_estimate_tokens(chunk_text),
                    metadata=metadata,
                ))
                idx += 1
            start += self.chunk_size - self.overlap
        return chunks


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 100) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,
            chunk_overlap=overlap * 4,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        splits = self.splitter.split_text(text)
        return [
            Chunk(
                content=s.strip(),
                index=i,
                token_count=_estimate_tokens(s),
                metadata=metadata,
            )
            for i, s in enumerate(splits) if s.strip()
        ]


class SemanticChunker(BaseChunker):
    def __init__(self, embedding_service: object, threshold: float = 0.5) -> None:
        self.embedding_service = embedding_service
        self.threshold = threshold
        self.min_chunk_size = 200 * 4  # minimum chars to avoid fragmentation

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return [Chunk(
                content=text, index=0, token_count=_estimate_tokens(text), metadata=metadata
            )]

        from app.ingestion.embedder import EmbeddingService
        embedder: EmbeddingService = self.embedding_service  # type: ignore[assignment]
        embeddings = embedder.embed_texts(sentences)

        chunks = []
        current_sentences: list[str] = [sentences[0]]
        idx = 0

        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            current_text = " ".join(current_sentences)

            if similarity < self.threshold and len(current_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=current_text.strip(),
                    index=idx,
                    token_count=_estimate_tokens(current_text),
                    metadata=metadata,
                ))
                idx += 1
                current_sentences = []

            current_sentences.append(sentences[i])

        if current_sentences:
            final_text = " ".join(current_sentences).strip()
            if final_text:
                chunks.append(Chunk(
                    content=final_text,
                    index=idx,
                    token_count=_estimate_tokens(final_text),
                    metadata=metadata,
                ))

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


STRATEGY_MAP = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
}


def get_chunker(strategy: str, **kwargs: object) -> BaseChunker:
    chunker_cls = STRATEGY_MAP.get(strategy)
    if chunker_cls is None:
        raise ValueError(f"unknown chunking strategy: {strategy}")
    return chunker_cls(**kwargs)  # type: ignore[arg-type]
