import httpx
import pytest


@pytest.fixture
def client():
    from expenso_assistant.api.main import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_health_is_green(client):
    async with client:
        response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mcp_enabled"] is True
