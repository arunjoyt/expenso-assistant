import json
import types

import httpx
import pytest
import respx
from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from fastmcp.exceptions import ToolError

from expenso_assistant import mcp_server, tools
from expenso_assistant.mcp_server import MemberContextMiddleware, build_mcp_server

from .conftest import API, method_url


async def test_mcp_exposes_every_tool_definition():
    async with Client(build_mcp_server(), mode="legacy") as client:
        names = {tool.name for tool in await client.list_tools()}

    assert names == {fn.__name__ for fn in (*tools.READ_TOOLS, *tools.WRITE_TOOLS)}


def test_mcp_not_mounted_when_disabled(monkeypatch):
    monkeypatch.setenv("MCP_ENABLED", "false")
    from expenso_assistant.config import get_settings

    get_settings.cache_clear()
    from expenso_assistant.api.main import create_app

    app = create_app()
    paths = {str(getattr(r, "path", "")) for r in app.routes}
    assert "/health" in paths
    assert not any(p.startswith("/mcp") for p in paths)


@respx.mock
async def test_write_tool_elicits_once_then_writes_with_connector_provenance():
    route = respx.post(method_url(f"{API}.create_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "e-new"}})
    )
    prompts = []

    async def accept(message, response_type, params, context):
        prompts.append(message)
        return True

    async with Client(build_mcp_server(), elicitation_handler=accept, mode="legacy") as client:
        result = await client.call_tool("create_expense", {"amount": 9, "notes": "Coffee"})

    assert len(prompts) == 1
    assert route.called
    assert json.loads(route.calls.last.request.read())["entry_method"] == "connector"
    assert result.data["name"] == "e-new"


@respx.mock
async def test_write_tool_does_not_write_on_decline():
    route = respx.post(method_url(f"{API}.create_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "nope"}})
    )

    async def decline(message, response_type, params, context):
        return ElicitResult(action="decline")

    async with Client(build_mcp_server(), elicitation_handler=decline, mode="legacy") as client:
        result = await client.call_tool("create_expense", {"amount": 9})

    assert not route.called
    assert result.data["status"] == "cancelled"


async def test_read_tool_needs_no_confirmation():
    with respx.mock:
        route = respx.get(method_url(f"{API}.list_categories")).mock(
            return_value=httpx.Response(200, json={"message": ["Groceries"]})
        )

        async def refuse(message, response_type, params, context):  # would fail the call
            return ElicitResult(action="decline")

        async with Client(build_mcp_server(), elicitation_handler=refuse, mode="legacy") as client:
            result = await client.call_tool("list_categories", {})

    assert route.called
    assert result.data == ["Groceries"]


async def test_write_rejected_before_elicitation_without_write_scope(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "get_settings", lambda: types.SimpleNamespace(frappe_oauth_client_id="x")
    )
    monkeypatch.setattr(
        mcp_server,
        "get_access_token",
        lambda: types.SimpleNamespace(token="t", scopes=["expenso:read"]),
    )

    called = False

    async def call_next(_):
        nonlocal called
        called = True

    ctx = types.SimpleNamespace(message=types.SimpleNamespace(name="create_expense"))

    with pytest.raises(ToolError, match="expenso:write"):
        await MemberContextMiddleware().on_call_tool(ctx, call_next)
    assert called is False
