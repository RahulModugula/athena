"""Redis-based caching for embeddings and retrieval results."""

import contextlib
import hashlib
import json
from typing import Any

import structlog

logger = structlog.get_logger()


class CacheService:
    def __init__(self, redis_url: str) -> None:
        self._url = redis_url
        self._client: Any = None

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(self._url, decode_responses=False)
            await self._client.ping()
            logger.info("cache connected", url=self._url)
        except Exception as e:
            logger.warning("cache unavailable", error=str(e))
            self._client = None

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    async def get(self, key: str) -> Any | None:
        if not self._client:
            return None
        try:
            data = await self._client.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        if not self._client:
            return
        with contextlib.suppress(Exception):
            await self._client.set(key, json.dumps(value), ex=ttl)

    @staticmethod
    def make_key(prefix: str, *parts: str) -> str:
        h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        return f"athena:{prefix}:{h}"
