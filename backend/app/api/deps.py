from fastapi import Request

from app.ingestion.embedder import EmbeddingService
from app.retrieval.reranker import RerankerService


def get_embedder(request: Request) -> EmbeddingService:
    return request.app.state.embedder


def get_reranker(request: Request) -> RerankerService:
    return request.app.state.reranker
