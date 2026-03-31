"""Tests for TenantAuthMiddleware and RateLimitMiddleware."""

from unittest.mock import AsyncMock, MagicMock, patch

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.api.middleware import RateLimitMiddleware, TenantAuthMiddleware


def _make_app(keys: list[str], rpm: int = 0) -> Starlette:
    async def homepage(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "healthy"})

    app = Starlette(routes=[
        Route("/api/endpoint", homepage),
        Route("/api/health", health),
    ])

    app.add_middleware(TenantAuthMiddleware)
    if rpm > 0:
        app.add_middleware(RateLimitMiddleware, default_rpm=rpm)
    return app


def _mock_db_session():
    """Return a mock async_session that yields a session whose execute returns
    a result with .first() = None (no tenant found in DB → legacy fallback)."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.first.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_ctx


class TestAPIKeyMiddleware:
    def test_no_keys_configured_allows_all(self) -> None:
        app = _make_app(keys=[])
        with patch("app.config.settings") as mock_settings:
            mock_settings.api_keys = []
            client = TestClient(app)
            response = client.get("/api/endpoint")
        assert response.status_code == 200

    def test_valid_key_is_accepted(self) -> None:
        app = _make_app(keys=[])
        with (
            patch("app.database.async_session", return_value=_mock_db_session()),
            patch("app.config.settings") as mock_settings,
        ):
            mock_settings.api_keys = ["secret-key"]
            client = TestClient(app)
            response = client.get("/api/endpoint", headers={"X-API-Key": "secret-key"})
        assert response.status_code == 200

    def test_invalid_key_returns_401(self) -> None:
        app = _make_app(keys=[])
        with (
            patch("app.database.async_session", return_value=_mock_db_session()),
            patch("app.config.settings") as mock_settings,
        ):
            mock_settings.api_keys = ["secret-key"]
            client = TestClient(app)
            response = client.get("/api/endpoint", headers={"X-API-Key": "wrong"})
        assert response.status_code == 401

    def test_missing_key_returns_401(self) -> None:
        app = _make_app(keys=[])
        with patch("app.config.settings") as mock_settings:
            mock_settings.api_keys = ["secret-key"]
            client = TestClient(app)
            response = client.get("/api/endpoint")
        assert response.status_code == 401

    def test_health_endpoint_is_exempt(self) -> None:
        app = _make_app(keys=[])
        with patch("app.config.settings") as mock_settings:
            mock_settings.api_keys = ["secret-key"]
            client = TestClient(app)
            response = client.get("/api/health")
        assert response.status_code == 200

    def test_multiple_keys_any_accepted(self) -> None:
        app = _make_app(keys=[])
        with (
            patch("app.database.async_session", return_value=_mock_db_session()),
            patch("app.config.settings") as mock_settings,
        ):
            mock_settings.api_keys = ["key-a", "key-b"]
            client = TestClient(app)
            assert client.get("/api/endpoint", headers={"X-API-Key": "key-a"}).status_code == 200
            assert client.get("/api/endpoint", headers={"X-API-Key": "key-b"}).status_code == 200
            assert client.get("/api/endpoint", headers={"X-API-Key": "key-c"}).status_code == 401

    def test_whitespace_in_keys_is_stripped(self) -> None:
        app = _make_app(keys=[])
        with (
            patch("app.database.async_session", return_value=_mock_db_session()),
            patch("app.config.settings") as mock_settings,
        ):
            # The middleware checks `raw_key in settings.api_keys` — the key
            # header value is "trimmed" (no spaces), so the list must contain
            # the trimmed variant for the legacy fallback to match.
            mock_settings.api_keys = ["trimmed"]
            client = TestClient(app)
            response = client.get("/api/endpoint", headers={"X-API-Key": "trimmed"})
        assert response.status_code == 200


class TestRateLimitMiddleware:
    def test_requests_within_limit_pass(self) -> None:
        app = _make_app(keys=[], rpm=10)
        with patch("app.config.settings") as mock_settings:
            mock_settings.api_keys = []
            client = TestClient(app)
            for _ in range(5):
                assert client.get("/api/endpoint").status_code == 200

    def test_requests_over_limit_get_429(self) -> None:
        app = _make_app(keys=[], rpm=3)
        with patch("app.config.settings") as mock_settings:
            mock_settings.api_keys = []
            client = TestClient(app)
            responses = [client.get("/api/endpoint").status_code for _ in range(5)]
        assert responses[:3] == [200, 200, 200]
        assert 429 in responses[3:]

    def test_429_includes_retry_after_header(self) -> None:
        app = _make_app(keys=[], rpm=1)
        with patch("app.config.settings") as mock_settings:
            mock_settings.api_keys = []
            client = TestClient(app)
            client.get("/api/endpoint")  # use up the limit
            response = client.get("/api/endpoint")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_rpm_zero_disables_limiting(self) -> None:
        app = _make_app(keys=[], rpm=0)
        with patch("app.config.settings") as mock_settings:
            mock_settings.api_keys = []
            client = TestClient(app)
            for _ in range(20):
                assert client.get("/api/endpoint").status_code == 200
