
import pytest

from app.ingestion.chunker import FixedSizeChunker, RecursiveChunker


@pytest.fixture
def sample_text() -> str:
    return (
        "Machine learning is a subset of artificial intelligence that focuses on "
        "building systems that learn from data. Unlike traditional programming where "
        "rules are explicitly coded, machine learning algorithms identify patterns "
        "in data and make decisions with minimal human intervention. "
        "Deep learning is a further subset of machine learning that uses neural "
        "networks with many layers. These deep neural networks can automatically "
        "discover representations from raw data. This eliminates the need for "
        "manual feature engineering. "
        "Reinforcement learning is another paradigm where agents learn by interacting "
        "with an environment. The agent receives rewards or penalties based on its "
        "actions and learns to maximize cumulative reward over time."
    )


@pytest.fixture
def fixed_chunker() -> FixedSizeChunker:
    return FixedSizeChunker(chunk_size=64, overlap=16)


@pytest.fixture
def recursive_chunker() -> RecursiveChunker:
    return RecursiveChunker(chunk_size=64, overlap=16)
