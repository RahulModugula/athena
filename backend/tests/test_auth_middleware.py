"""Tests for APIKeyMiddleware and RateLimitMiddleware."""

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.api.middleware import APIKeyMiddleware, RateLimitMiddleware


def _make_app(keys: list[str], rpm: int = 0) -> Starlette:
    async def homepage(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "healthy"})

    app = Starlette(routes=[
        Route("/api/endpoint", homepage),
        Route("/api/health", health),
    ])
    app.add_middleware(APIKeyMiddleware, api_keys=keys)
    if rpm > 0:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=rpm)
    return app


class TestAPIKeyMiddleware:
    def test_no_keys_configured_allows_all(self) -> None:
        client = TestClient(_make_app(keys=[]))
        response = client.get("/api/endpoint")
        assert response.status_code == 200

    def test_valid_key_is_accepted(self) -> None:
        client = TestClient(_make_app(keys=["secret-key"]))
        response = client.get("/api/endpoint", headers={"X-API-Key": "secret-key"})
        assert response.status_code == 200

    def test_invalid_key_returns_401(self) -> None:
        client = TestClient(_make_app(keys=["secret-key"]))
        response = client.get("/api/endpoint", headers={"X-API-Key": "wrong"})
        assert response.status_code == 401
        assert "X-API-Key" in response.json()["detail"]

    def test_missing_key_returns_401(self) -> None:
        client = TestClient(_make_app(keys=["secret-key"]))
        response = client.get("/api/endpoint")
        assert response.status_code == 401

    def test_health_endpoint_is_exempt(self) -> None:
        client = TestClient(_make_app(keys=["secret-key"]))
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_multiple_keys_any_accepted(self) -> None:
        client = TestClient(_make_app(keys=["key-a", "key-b"]))
        assert client.get("/api/endpoint", headers={"X-API-Key": "key-a"}).status_code == 200
        assert client.get("/api/endpoint", headers={"X-API-Key": "key-b"}).status_code == 200
        assert client.get("/api/endpoint", headers={"X-API-Key": "key-c"}).status_code == 401

    def test_whitespace_in_keys_is_stripped(self) -> None:
        client = TestClient(_make_app(keys=["  trimmed  "]))
        response = client.get("/api/endpoint", headers={"X-API-Key": "trimmed"})
        assert response.status_code == 200


class TestRateLimitMiddleware:
    def test_requests_within_limit_pass(self) -> None:
        client = TestClient(_make_app(keys=[], rpm=10))
        for _ in range(5):
            assert client.get("/api/endpoint").status_code == 200

    def test_requests_over_limit_get_429(self) -> None:
        client = TestClient(_make_app(keys=[], rpm=3))
        responses = [client.get("/api/endpoint").status_code for _ in range(5)]
        assert responses[:3] == [200, 200, 200]
        assert 429 in responses[3:]

    def test_429_includes_retry_after_header(self) -> None:
        client = TestClient(_make_app(keys=[], rpm=1))
        client.get("/api/endpoint")  # use up the limit
        response = client.get("/api/endpoint")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_rpm_zero_disables_limiting(self) -> None:
        client = TestClient(_make_app(keys=[], rpm=0))
        for _ in range(20):
            assert client.get("/api/endpoint").status_code == 200
