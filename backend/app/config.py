from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "ATHENA_"}

    database_url: str = "postgresql+asyncpg://athena:athena@localhost:5432/athena"
    database_url_sync: str = "postgresql://athena:athena@localhost:5432/athena"

    llm_provider: str = "anthropic"  # "anthropic" or "zhipuai"
    anthropic_api_key: str = ""
    zhipuai_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    # For ZhipuAI, use: glm-z1-air, glm-4-flash, glm-4-plus
    zhipuai_model: str = "glm-z1-air"

    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 512
    chunk_overlap: int = 100
    top_k: int = 10
    rerank_top_k: int = 5
    rrf_k: int = 60

    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:8501", "http://localhost:5173"]


settings = Settings()
