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


class DocsChunker(BaseChunker):
    """Documentation-aware chunker that preserves code blocks and heading hierarchy.

    Splits Markdown/HTML documentation by headings, keeps fenced code blocks
    intact, and attaches heading path metadata to each chunk for richer search.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 100) -> None:
        self.max_chars = chunk_size * 4
        self.overlap_chars = overlap * 4
        # Fallback splitter for oversized prose sections
        self._fallback = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chars,
            chunk_overlap=self.overlap_chars,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        import re

        metadata = metadata or {}
        sections = self._split_by_headings(text)
        chunks: list[Chunk] = []
        idx = 0

        for heading_path, section_text in sections:
            # Split section into code blocks and prose segments
            parts = re.split(r"(```[\s\S]*?```)", section_text)
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                is_code = part.startswith("```") and part.endswith("```")
                language = ""
                if is_code:
                    first_line = part.split("\n", 1)[0]
                    language = first_line.removeprefix("```").strip()

                chunk_meta = {
                    **metadata,
                    "heading_path": heading_path,
                    "has_code": is_code,
                }
                if language:
                    chunk_meta["language"] = language

                if len(part) <= self.max_chars:
                    chunks.append(Chunk(
                        content=part, index=idx,
                        token_count=_estimate_tokens(part), metadata=chunk_meta,
                    ))
                    idx += 1
                elif is_code:
                    # Split large code blocks by blank lines / function boundaries
                    for sub in self._split_large_code(part):
                        chunks.append(Chunk(
                            content=sub, index=idx,
                            token_count=_estimate_tokens(sub), metadata=chunk_meta,
                        ))
                        idx += 1
                else:
                    for sub in self._fallback.split_text(part):
                        sub = sub.strip()
                        if sub:
                            chunks.append(Chunk(
                                content=sub, index=idx,
                                token_count=_estimate_tokens(sub), metadata=chunk_meta,
                            ))
                            idx += 1

        return chunks

    @staticmethod
    def _split_by_headings(text: str) -> list[tuple[str, str]]:
        """Split text into (heading_path, section_body) pairs."""
        import re

        heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(text))

        if not matches:
            return [("", text)]

        sections = []
        heading_stack: list[tuple[int, str]] = []

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()

            # Pop headings of same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            heading_path = " > ".join(h[1] for _, h in enumerate(heading_stack))
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((heading_path, body))

        # Include any text before the first heading
        if matches and matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.insert(0, ("", preamble))

        return sections

    def _split_large_code(self, code_block: str) -> list[str]:
        """Split a large fenced code block into smaller pieces."""
        lines = code_block.split("\n")
        # Remove fences
        if lines and lines[0].startswith("```"):
            fence_open = lines[0]
            lines = lines[1:]
        else:
            fence_open = "```"
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        pieces: list[str] = []
        current: list[str] = []
        current_len = 0

        for line in lines:
            current.append(line)
            current_len += len(line) + 1
            if current_len >= self.max_chars - 100:
                pieces.append(fence_open + "\n" + "\n".join(current) + "\n```")
                current = []
                current_len = 0

        if current:
            pieces.append(fence_open + "\n" + "\n".join(current) + "\n```")

        return pieces


STRATEGY_MAP = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
    "docs": DocsChunker,
}


def get_chunker(strategy: str, **kwargs: object) -> BaseChunker:
    chunker_cls = STRATEGY_MAP.get(strategy)
    if chunker_cls is None:
        raise ValueError(f"unknown chunking strategy: {strategy}")
    return chunker_cls(**kwargs)  # type: ignore[arg-type]
