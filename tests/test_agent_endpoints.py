"""The FastAPI Assistant endpoints: auth, thread isolation, SSE."""

from __future__ import annotations

import json
import urllib.parse

import httpx
import pytest
import respx
from langgraph.checkpoint.memory import InMemorySaver

from expenso_assistant import tools
from expenso_assistant.agent.graph import build_graph
from expenso_assistant.config import get_settings

from .conftest import FRAPPE_URL
from .fakes import ScriptedChatModel
from .test_agent_session import answer

INTROSPECT = f"{FRAPPE_URL}/api/method/frappe.integrations.oauth2.introspect_token"
LOGGED_USER = f"{FRAPPE_URL}/api/method/frappe.auth.get_logged_user"

# token -> (email, scope)
TOKENS = {
    "tok-a": ("amir@example.com", "all openid expenso:read"),
    "tok-b": ("bianca@example.com", "all openid expenso:read"),
    "tok-readless": ("carl@example.com", "all openid"),
}


@pytest.fixture(autouse=True)
def _no_mcp(monkeypatch):
    monkeypatch.setenv("MCP_ENABLED", "false")
    get_settings.cache_clear()


@pytest.fixture
def app(spy_langfuse):
    spy_langfuse()
    model = ScriptedChatModel(responses=[answer("You spent 12.")] * 4)
    checkpointer = InMemorySaver()
    from expenso_assistant.api.main import create_app

    graph = build_graph(model, checkpointer=checkpointer)
    return create_app(graph=graph, checkpointer=checkpointer)


@pytest.fixture
def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _wire_frappe() -> None:
    def introspect(request: httpx.Request) -> httpx.Response:
        form = dict(urllib.parse.parse_qsl(request.content.decode()))
        entry = TOKENS.get(form.get("token"))
        if entry is None:
            return httpx.Response(200, json={"active": False})
        return httpx.Response(200, json={"active": True, "scope": entry[1], "exp": 9999999999})

    def logged_user(request: httpx.Request) -> httpx.Response:
        token = request.headers["authorization"].removeprefix("Bearer ")
        return httpx.Response(200, json={"message": TOKENS[token][0]})

    respx.post(INTROSPECT).mock(side_effect=introspect)
    respx.get(LOGGED_USER).mock(side_effect=logged_user)


def _events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        head, data = block.split("\n", 1)
        out.append((head.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


async def test_every_endpoint_401s_without_a_valid_bearer(client):
    async with client:
        assert (await client.post("/chat", json={"message": "hi"})).status_code == 401
        assert (await client.get("/history")).status_code == 401
        assert (await client.post("/resume", json={})).status_code == 401
        assert (await client.delete("/history")).status_code == 401
        bad = {"Authorization": "Bearer nonsense"}
        with respx.mock:
            _wire_frappe()
            assert (await client.get("/history", headers=bad)).status_code == 401


async def test_read_scope_is_required(client):
    async with client, respx.mock:
        _wire_frappe()
        r = await client.get("/history", headers={"Authorization": "Bearer tok-readless"})
    assert r.status_code == 401


async def test_chat_streams_sse_and_persists_to_the_members_own_thread(client):
    async with client, respx.mock:
        _wire_frappe()
        chat = await client.post(
            "/chat", json={"message": "spend?"}, headers={"Authorization": "Bearer tok-a"}
        )
        history = await client.get("/history", headers={"Authorization": "Bearer tok-a"})

    assert chat.headers["content-type"].startswith("text/event-stream")
    kinds = [kind for kind, _ in _events(chat.text)]
    assert kinds[-1] == "done" and "token" in kinds
    assert [m["content"] for m in history.json()["messages"]] == ["spend?", "You spent 12."]


async def test_one_member_cannot_see_anothers_thread(client):
    async with client, respx.mock:
        _wire_frappe()
        await client.post(
            "/chat", json={"message": "amir's question"}, headers={"Authorization": "Bearer tok-a"}
        )
        bianca = await client.get("/history", headers={"Authorization": "Bearer tok-b"})

    assert bianca.json()["messages"] == []


async def test_chat_accepts_an_attached_image(client):
    async with client, respx.mock:
        _wire_frappe()
        chat = await client.post(
            "/chat",
            json={"message": "", "image": "data:image/jpeg;base64,Zm9v"},
            headers={"Authorization": "Bearer tok-a"},
        )
    assert chat.status_code == 200
    assert _events(chat.text)[0] == ("step", {"text": "Reading the receipt…"})


async def test_chat_rejects_an_oversized_image(monkeypatch, spy_langfuse):
    # Settings are captured by `create_app` at construction time, so the cap
    # must be set *before* building the app — the shared `app`/`client`
    # fixtures already baked in the default cap by the time a test body runs.
    from expenso_assistant.api.main import create_app
    from expenso_assistant.config import get_settings

    monkeypatch.setenv("MAX_RECEIPT_IMAGE_CHARS", "10")
    get_settings.cache_clear()
    spy_langfuse()
    checkpointer = InMemorySaver()
    graph = build_graph(ScriptedChatModel(responses=[]), checkpointer=checkpointer)
    app = create_app(graph=graph, checkpointer=checkpointer)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async with client, respx.mock:
        _wire_frappe()
        chat = await client.post(
            "/chat",
            json={"message": "", "image": "data:image/jpeg;base64,Zm9vYmFyYmF6"},
            headers={"Authorization": "Bearer tok-a"},
        )
    assert chat.status_code == 413


async def test_chat_rejects_an_oversized_message(monkeypatch, spy_langfuse):
    # Same caveat as the oversized-image test above: the cap is captured by
    # `create_app` at construction time, so it must be set before building it.
    from expenso_assistant.api.main import create_app
    from expenso_assistant.config import get_settings

    monkeypatch.setenv("MAX_CHAT_MESSAGE_CHARS", "10")
    get_settings.cache_clear()
    spy_langfuse()
    checkpointer = InMemorySaver()
    graph = build_graph(ScriptedChatModel(responses=[]), checkpointer=checkpointer)
    app = create_app(graph=graph, checkpointer=checkpointer)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async with client, respx.mock:
        _wire_frappe()
        chat = await client.post(
            "/chat",
            json={"message": "this message is way over the ten character cap"},
            headers={"Authorization": "Bearer tok-a"},
        )
    assert chat.status_code == 413


async def test_clear_chat_empties_the_thread(client):
    async with client, respx.mock:
        _wire_frappe()
        head = {"Authorization": "Bearer tok-a"}
        await client.post("/chat", json={"message": "hi"}, headers=head)
        cleared = await client.delete("/history", headers=head)
        after = await client.get("/history", headers=head)

    assert cleared.json() == {"status": "cleared"}
    assert after.json()["messages"] == []


# --- /run/proactive (P7-S2) ------------------------------------------------


@pytest.fixture
def proactive_app(spy_langfuse):
    spy_langfuse()
    checkpointer = InMemorySaver()
    graph = build_graph(ScriptedChatModel(responses=[]), checkpointer=checkpointer)
    read_only_graph = build_graph(
        ScriptedChatModel(responses=[answer("Here is your insight.")]),
        checkpointer=checkpointer,
        tools=tools.READ_TOOLS,
    )
    from expenso_assistant.api.main import create_app

    return create_app(graph=graph, checkpointer=checkpointer, read_only_graph=read_only_graph)


@pytest.fixture
def proactive_client(proactive_app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proactive_app), base_url="http://test"
    )


async def test_run_proactive_returns_202_and_the_insight_lands_in_history(proactive_client):
    async with proactive_client, respx.mock:
        _wire_frappe()
        head = {"Authorization": "Bearer tok-a"}
        response = await proactive_client.post(
            "/run/proactive", json={"job": "monthly_summary"}, headers=head
        )
        history = await proactive_client.get("/history", headers=head)

    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}
    messages = history.json()["messages"]
    assert len(messages) == 1
    assert messages[0]["kind"] == "insight"
    assert messages[0]["content"] == "Here is your insight."


async def test_run_proactive_requires_a_valid_bearer(proactive_client):
    async with proactive_client:
        response = await proactive_client.post("/run/proactive", json={"job": "monthly_summary"})
    assert response.status_code == 401
