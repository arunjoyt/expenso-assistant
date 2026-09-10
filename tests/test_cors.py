"""CORS preflight for the in-app Assistant tab (P6-S6 companion change).

The tab calls `/chat` from the Frappe origin with an `Authorization` header, so
a preflight from an allowed origin must come back with the CORS headers and a
disallowed origin must not.
"""

from __future__ import annotations

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from expenso_assistant.agent.graph import build_graph
from expenso_assistant.config import get_settings

from .fakes import ScriptedChatModel

ALLOWED = "https://expenso.example.com"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setenv("ALLOWED_CORS_ORIGINS", f"{ALLOWED}, https://other.example.com")
    get_settings.cache_clear()

    from expenso_assistant.api.main import create_app

    checkpointer = InMemorySaver()
    graph = build_graph(ScriptedChatModel(responses=[]), checkpointer=checkpointer)
    app = create_app(graph=graph, checkpointer=checkpointer)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _preflight(origin: str) -> dict:
    return {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    }


async def test_preflight_from_an_allowed_origin_gets_the_cors_headers(client):
    async with client:
        r = await client.options("/chat", headers=_preflight(ALLOWED))

    assert r.headers["access-control-allow-origin"] == ALLOWED
    allow_headers = r.headers["access-control-allow-headers"].lower()
    assert "authorization" in allow_headers
    assert "POST" in r.headers["access-control-allow-methods"]


async def test_preflight_from_a_disallowed_origin_gets_no_cors_headers(client):
    async with client:
        r = await client.options("/chat", headers=_preflight("https://evil.example.com"))

    assert "access-control-allow-origin" not in r.headers
