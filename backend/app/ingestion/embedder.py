import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        logger.info("loading embedding model", model=model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("embedding model loaded", dimension=self.dimension)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(
            f"Represent this sentence for searching relevant passages: {query}",
            normalize_embeddings=True,
        )
        return embedding.tolist()
