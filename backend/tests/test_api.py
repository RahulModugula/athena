
import pytest


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_status(self) -> None:
        from app.models.schemas import HealthResponse

        response = HealthResponse(
            embedding_model="test-model",
            llm_model="glm-4",
            document_count=0,
        )
        assert response.status == "healthy"
        assert response.version == "0.1.0"
        assert response.document_count == 0
