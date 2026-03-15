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

    await log.ainfo("models loaded")
    yield
    await log.ainfo("shutting down athena")


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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log = structlog.get_logger()
    await log.aerror("unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


from app.api.routes import router  # noqa: E402

app.include_router(router, prefix="/api")
