import structlog
from sentence_transformers import CrossEncoder

logger = structlog.get_logger()


class RerankerService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        logger.info("loading reranker model", model=model_name)
        self.model = CrossEncoder(model_name)
        logger.info("reranker model loaded")

    def rerank(
        self, query: str, texts: list[str], top_k: int = 5
    ) -> list[tuple[int, float]]:
        pairs = [(query, text) for text in texts]
        scores = self.model.predict(pairs)
        indexed_scores = list(enumerate(scores.tolist()))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]
