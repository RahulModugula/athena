from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.observability.logging import configure_logging

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    log = structlog.get_logger()
    await log.ainfo("starting athena", embedding_model=settings.embedding_model)

    from app.ingestion.embedder import EmbeddingService
    from app.retrieval.reranker import RerankerService

    app.state.embedder = EmbeddingService(settings.embedding_model)
    app.state.reranker = RerankerService(settings.reranker_model)

    if settings.graph_rag_enabled and settings.neo4j_uri:
        from app.graph.store import GraphStore

        graph_store = GraphStore(
            uri=settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await graph_store.connect()
        app.state.graph_store = graph_store
    else:
        app.state.graph_store = None

    import app.mcp._store_ref as _store_ref
    from app.database import async_session as _session_factory

    _store_ref.graph_store = app.state.graph_store
    _store_ref.embedder = app.state.embedder
    _store_ref.reranker = app.state.reranker
    _store_ref.db_session_factory = _session_factory

    if settings.cache_enabled and settings.redis_url:
        from app.services.cache import CacheService

        app.state.cache = CacheService(settings.redis_url)
        await app.state.cache.connect()
    else:
        app.state.cache = None

    await log.ainfo("models loaded")
    yield
    await log.ainfo("shutting down athena")
    if app.state.graph_store is not None:
        await app.state.graph_store.close()
    if app.state.cache is not None:
        await app.state.cache.close()


app = FastAPI(
    title="Athena",
    description="RAG-powered research assistant with hybrid search",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.middleware import APIKeyMiddleware, RateLimitMiddleware  # noqa: E402

app.add_middleware(APIKeyMiddleware, api_keys=settings.api_keys)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log = structlog.get_logger()
    await log.aerror("unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


from prometheus_client import make_asgi_app  # noqa: E402

from app.api.routes import router  # noqa: E402
from app.mcp.server import mcp  # noqa: E402

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
app.include_router(router, prefix="/api")
app.mount("/mcp", mcp.sse_app())
