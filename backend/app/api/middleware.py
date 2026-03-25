"""Request middleware: per-tenant API key authentication and rate limiting."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from sqlalchemy import select, update
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger()

# Paths that bypass authentication entirely
_AUTH_EXEMPT = frozenset({
    "/api/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/metrics",
})

# Rate limit tiers: plan → requests per minute
PLAN_LIMITS: dict[str, int] = {
    "free": 10,
    "starter": 100,
    "pro": 500,
    "enterprise": 0,  # 0 = unlimited
}


def _hash_key(raw_key: str) -> str:
    """Hash an API key for storage/lookup using SHA-256."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


class TenantAuthMiddleware(BaseHTTPMiddleware):
    """Resolve API key → tenant and set request.state.tenant_id.

    Supports two key types:
    - ``sk_*`` (secret): full API access for dashboard / server-to-server
    - ``pk_*`` (publishable): widget query endpoints only

    When no keys are configured in the database at all (fresh install), the
    middleware falls back to the legacy ``settings.api_keys`` list so existing
    setups keep working.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path

        # Always allow exempt paths
        if any(path == p or path.startswith(p + "/") for p in _AUTH_EXEMPT):
            return await call_next(request)

        # MCP and widget endpoints handle auth separately
        if path.startswith("/mcp"):
            return await call_next(request)

        raw_key = (
            request.headers.get("X-API-Key", "")
            or request.headers.get("X-Widget-Key", "")
        )

        if not raw_key:
            # Allow unauthenticated requests if the system has no tenants yet
            # (backwards-compatible with single-user Athena)
            from app.config import settings
            if not settings.api_keys:
                request.state.tenant_id = None
                request.state.tenant = None
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content={"detail": "missing API key — pass X-API-Key header"},
            )

        # Look up key in database
        from app.database import async_session
        from app.models.orm import APIKey, Tenant

        key_hash = _hash_key(raw_key)

        async with async_session() as session:
            result = await session.execute(
                select(APIKey, Tenant)
                .join(Tenant, APIKey.tenant_id == Tenant.id)
                .where(APIKey.key_hash == key_hash, APIKey.is_active.is_(True))
            )
            row = result.first()

            if row is None:
                # Fall back to legacy static key check
                from app.config import settings
                if raw_key in settings.api_keys:
                    request.state.tenant_id = None
                    request.state.tenant = None
                    return await call_next(request)

                return JSONResponse(
                    status_code=401,
                    content={"detail": "invalid API key"},
                )

            api_key_obj, tenant = row

            # Widget endpoints only accept publishable keys
            if path.startswith("/api/widget") and api_key_obj.key_type != "publishable":
                return JSONResponse(
                    status_code=403,
                    content={"detail": "use a publishable key (pk_*) for widget endpoints"},
                )

            request.state.tenant_id = tenant.id
            request.state.tenant = tenant
            request.state.api_key_type = api_key_obj.key_type

            # Update last_used_at (fire-and-forget)
            await session.execute(
                update(APIKey)
                .where(APIKey.id == api_key_obj.id)
                .values(last_used_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            )
            await session.commit()

        return await call_next(request)


async def _redis_is_rate_limited(
    redis_client: Any, limit_key: str, rpm: int, now: float
) -> tuple[bool, int]:
    """Check and update rate limit using a Redis sorted set (sliding window).

    Returns (is_limited, retry_after_seconds).
    """
    window_key = f"athena:rl:{limit_key}"
    cutoff = now - 60.0
    pipeline = redis_client.pipeline()
    pipeline.zremrangebyscore(window_key, "-inf", cutoff)
    pipeline.zcard(window_key)
    pipeline.zadd(window_key, {str(now): now})
    pipeline.expire(window_key, 61)
    results = await pipeline.execute()
    count_before_add = results[1]
    if count_before_add >= rpm:
        # Remove the timestamp we just added since we're rejecting
        await redis_client.zrem(window_key, str(now))
        return True, 61
    return False, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by tenant (or IP for unauthenticated).

    Uses Redis when available (multi-replica safe) with in-process deque fallback.
    """

    def __init__(self, app, default_rpm: int = 60) -> None:
        super().__init__(app)
        self._default_rpm = default_rpm
        # In-process fallback when Redis is unavailable
        self._windows: defaultdict[str, deque[float]] = defaultdict(deque)

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the last IP added by a trusted proxy, not the first (spoofable)
            parts = [p.strip() for p in forwarded.split(",")]
            return parts[-1] if parts else "unknown"
        return request.client.host if request.client else "unknown"

    def _in_memory_check(self, limit_key: str, rpm: int, now: float) -> tuple[bool, int]:
        window = self._windows[limit_key]
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= rpm:
            retry_after = int(60 - (now - window[0])) + 1
            return True, retry_after
        window.append(now)
        return False, 0

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Determine rate limit key and rpm
        tenant = getattr(request.state, "tenant", None)
        if tenant is not None:
            limit_key = f"tenant:{tenant.id}"
            rpm = PLAN_LIMITS.get(tenant.plan, self._default_rpm)
        else:
            limit_key = f"ip:{self._client_ip(request)}"
            rpm = self._default_rpm

        if rpm <= 0:
            return await call_next(request)

        now = time.monotonic()
        is_limited = False
        retry_after = 60

        # Prefer Redis for distributed rate limiting; fall back to in-process
        cache = getattr(request.app.state, "cache", None)
        redis_client = getattr(cache, "_client", None) if cache else None
        if redis_client is not None:
            try:
                is_limited, retry_after = await _redis_is_rate_limited(
                    redis_client, limit_key, rpm, now
                )
            except Exception as exc:
                logger.warning("redis rate limit check failed, using in-memory", error=str(exc))
                is_limited, retry_after = self._in_memory_check(limit_key, rpm, now)
        else:
            is_limited, retry_after = self._in_memory_check(limit_key, rpm, now)

        if is_limited:
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
