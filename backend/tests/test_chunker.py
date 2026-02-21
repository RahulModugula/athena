import pytest

from app.ingestion.chunker import (
    FixedSizeChunker,
    RecursiveChunker,
    get_chunker,
)


class TestFixedSizeChunker:
    def test_produces_chunks(self, sample_text: str, fixed_chunker: FixedSizeChunker) -> None:
        chunks = fixed_chunker.chunk(sample_text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.content
            assert chunk.token_count > 0

    def test_chunk_indices_sequential(self, sample_text: str, fixed_chunker: FixedSizeChunker) -> None:
        chunks = fixed_chunker.chunk(sample_text)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_preserves_metadata(self, sample_text: str, fixed_chunker: FixedSizeChunker) -> None:
        meta = {"source": "test.pdf"}
        chunks = fixed_chunker.chunk(sample_text, metadata=meta)
        for chunk in chunks:
            assert chunk.metadata == meta

    def test_empty_text(self, fixed_chunker: FixedSizeChunker) -> None:
        chunks = fixed_chunker.chunk("")
        assert len(chunks) == 0


class TestRecursiveChunker:
    def test_produces_chunks(self, sample_text: str, recursive_chunker: RecursiveChunker) -> None:
        chunks = recursive_chunker.chunk(sample_text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.content
            assert chunk.token_count > 0

    def test_respects_separators(self, recursive_chunker: RecursiveChunker) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = recursive_chunker.chunk(text)
        assert len(chunks) >= 1

    def test_empty_text(self, recursive_chunker: RecursiveChunker) -> None:
        chunks = recursive_chunker.chunk("")
        assert len(chunks) == 0


class TestGetChunker:
    def test_fixed(self) -> None:
        chunker = get_chunker("fixed")
        assert isinstance(chunker, FixedSizeChunker)

    def test_recursive(self) -> None:
        chunker = get_chunker("recursive")
        assert isinstance(chunker, RecursiveChunker)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown chunking strategy"):
            get_chunker("nonexistent")
