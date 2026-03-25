from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {
        "env_prefix": "ATHENA_",
        "env_file": "../.env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    database_url: str = "postgresql+asyncpg://athena:athena@localhost:5432/athena"
    database_url_sync: str = "postgresql://athena:athena@localhost:5432/athena"

    llm_provider: str = "openrouter"  # "anthropic", "zhipuai", or "openrouter"
    anthropic_api_key: str = ""
    zhipuai_api_key: str = ""
    openrouter_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    # ZhipuAI models: glm-4.5-air, glm-4.5, glm-4.6, glm-4.7
    zhipuai_model: str = "glm-4.7"
    # OpenRouter free models: openai/gpt-oss-20b:free, meta-llama/llama-3.3-70b-instruct:free
    openrouter_model: str = "openai/gpt-oss-20b:free"

    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 512
    chunk_overlap: int = 100
    top_k: int = 10
    rerank_top_k: int = 5
    rrf_k: int = 60

    # Contextual Retrieval (Anthropic, Sep 2024): prepend an LLM-generated context
    # snippet to each chunk at ingestion time. Reduces top-20 retrieval failures by
    # ~49% at the cost of one LLM call per chunk (use a cheap/fast model).
    # Set contextual_retrieval_model to a fast model (e.g. claude-haiku-4-20250514)
    # to keep indexing costs low. Has no query-time overhead.
    contextual_retrieval_enabled: bool = False
    contextual_retrieval_model: str = "claude-haiku-4-20250514"

    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    graph_rag_enabled: bool = False

    redis_url: str = ""
    cache_enabled: bool = False

    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:8501", "http://localhost:5173"]

    # API key authentication — comma-separated list; empty string disables auth
    api_keys: list[str] = []
    # Rate limiting — requests per minute per IP (0 = disabled)
    rate_limit_per_minute: int = 60


settings = Settings()
