"""Request middleware: API key authentication and rate limiting."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger()

# Paths that bypass authentication (but not rate limiting)
_AUTH_EXEMPT = frozenset({
    "/api/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/metrics",
    "/mcp",
})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate X-API-Key header when api_keys are configured.

    If settings.api_keys is empty the middleware is a no-op, so local
    development works without any configuration.
    """

    def __init__(self, app, api_keys: list[str]) -> None:
        super().__init__(app)
        self._keys: frozenset[str] = frozenset(k.strip() for k in api_keys if k.strip())

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not self._keys:
            return await call_next(request)

        path = request.url.path
        if any(path == p or path.startswith(p + "/") for p in _AUTH_EXEMPT):
            return await call_next(request)

        key = request.headers.get("X-API-Key", "")
        if key not in self._keys:
            await logger.awarning("rejected request: invalid api key", path=path)
            return JSONResponse(
                status_code=401,
                content={"detail": "missing or invalid API key — pass X-API-Key header"},
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by client IP.

    Uses an in-process deque per IP — good enough for a single-replica
    deployment.  Set rate_limit_per_minute=0 to disable entirely.
    """

    def __init__(self, app, requests_per_minute: int) -> None:
        super().__init__(app)
        self._rpm = requests_per_minute
        # ip → deque of request timestamps (float seconds)
        self._windows: defaultdict[str, deque[float]] = defaultdict(deque)

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if self._rpm <= 0:
            return await call_next(request)

        ip = self._client_ip(request)
        now = time.monotonic()
        window = self._windows[ip]

        # Drop timestamps outside the 60-second window
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self._rpm:
            retry_after = int(60 - (now - window[0])) + 1
            await logger.awarning("rate limit exceeded", ip=ip, path=request.url.path)
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded — slow down"},
                headers={"Retry-After": str(retry_after)},
            )

        window.append(now)
        return await call_next(request)
