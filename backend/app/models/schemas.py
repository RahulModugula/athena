import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    DOCS = "docs"


class RetrievalStrategy(StrEnum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    mime_type: str
    metadata: dict[str, object] = Field(default_factory=dict)
    chunk_count: int = 0
    created_at: datetime


class ChunkResponse(BaseModel):
    id: uuid.UUID
    content: str
    chunk_index: int
    token_count: int
    chunking_strategy: str
    score: float = 0.0


class QueryRequest(BaseModel):
    question: str
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = Field(default=5, ge=1, le=50)
    document_ids: list[uuid.UUID] | None = None


class SourceChunk(BaseModel):
    chunk_id: uuid.UUID
    content: str
    document_name: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: float
    strategy: str


class SearchRequest(BaseModel):
    query: str
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = Field(default=10, ge=1, le=100)


class SearchResponse(BaseModel):
    chunks: list[ChunkResponse]
    strategy: str
    latency_ms: float


class EvalRequest(BaseModel):
    dataset: str = "sample_qa"
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID


class EvalMetrics(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float


class EvalRunResponse(BaseModel):
    id: uuid.UUID
    dataset_name: str
    chunking_strategy: str
    retrieval_strategy: str
    status: str = "completed"
    metrics: EvalMetrics
    sample_count: int
    created_at: datetime


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    embedding_model: str
    llm_model: str
    document_count: int


class AgentStep(BaseModel):
    agent: str
    action: str
    duration_ms: float


class FactCheckResult(BaseModel):
    claim: str
    supported: bool
    confidence: float
    evidence: list[str] = Field(default_factory=list)


class ResearchRequest(BaseModel):
    question: str
    max_iterations: int = Field(default=3, ge=1, le=5)
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID


class ResearchResponse(BaseModel):
    answer: str
    analysis: str
    fact_check: list[FactCheckResult] = Field(default_factory=list)
    sources: list[SourceChunk] = Field(default_factory=list)
    agent_trace: list[AgentStep] = Field(default_factory=list)
    latency_ms: float
