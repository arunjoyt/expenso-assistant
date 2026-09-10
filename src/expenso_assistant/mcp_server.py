"""FastMCP server — the external-connector adapter.

A **pure adapter**: it registers the `tools.py` functions for ChatGPT/Claude
connectors and gates every write behind a human confirmation. Mounted at `/mcp`
only when `MCP_ENABLED`; disabling it has zero effect on the in-app Assistant
(ADR 0008).

**Write confirmation — SEP-2322 guard pattern.** The MCP `2026-07-28` spec era
removed server-initiated elicitation, so a write tool does not push a prompt
mid-call. Instead, on the first invocation it returns an `InputRequiredResult`
("confirm this change?"); the connector renders its own confirm UI, then
re-invokes the tool with the answer, which the tool reads from
`ctx.input_responses` before doing the Frappe write. The *intent* — every
connector write is confirmed by the Member — is unchanged from ADR 0008's
"MCP elicitation on the writes"; only the wire mechanism moved.

Per tool call, middleware:
- binds a `FrappeClient` built from the caller's bearer, so `tools.py` reaches
  Frappe as that Member;
- binds `entry_method="connector"` for the writes;
- rejects a write tool when the token lacks `expenso:write` — before the
  confirm round-trip even starts.
"""

from __future__ import annotations

import functools

import mcp.types as mt
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_access_token, get_context
from fastmcp.server.middleware import Middleware, MiddlewareContext

from . import tools
from .auth import build_auth_provider
from .config import get_settings
from .frappe_client import FrappeClient, bind_frappe_client, reset_frappe_client

WRITE_SCOPE = "expenso:write"
_WRITE_TOOL_NAMES = frozenset(fn.__name__ for fn in tools.WRITE_TOOLS)

_CONFIRM_KEY = "confirm"
_CONFIRM_SCHEMA = {
    "type": "object",
    "properties": {_CONFIRM_KEY: {"type": "boolean", "description": "Apply this change?"}},
    "required": [_CONFIRM_KEY],
}


def _describe(name: str, arguments: dict | None) -> str:
    shown = {key: value for key, value in (arguments or {}).items() if value is not None}
    body = ", ".join(f"{key}={value!r}" for key, value in shown.items())
    return f"{name}({body})"


def _confirm_request(summary: str) -> mt.InputRequiredResult:
    return mt.InputRequiredResult(
        input_requests={
            _CONFIRM_KEY: mt.ElicitRequest(
                params=mt.ElicitRequestFormParams(
                    message=f"Confirm this change: {summary}",
                    requestedSchema=_CONFIRM_SCHEMA,
                )
            )
        },
        request_state=summary,
    )


def _confirmed(responses) -> bool:
    answer = responses.get(_CONFIRM_KEY) if responses else None
    return (
        answer is not None
        and getattr(answer, "action", None) == "accept"
        and bool((getattr(answer, "content", None) or {}).get(_CONFIRM_KEY))
    )


def _needs_write_confirmation(fn):
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        ctx = get_context()
        if ctx.input_responses is None:
            return _confirm_request(_describe(fn.__name__, kwargs))
        if not _confirmed(ctx.input_responses):
            return {"status": "cancelled", "proposed": ctx.request_state}
        return await fn(*args, **kwargs)

    return wrapper


class MemberContextMiddleware(Middleware):
    """Bind the Member's Frappe client + connector provenance for each tool call.

    Reads the caller's bearer from the validated OAuth access token. When MCP
    auth is switched off (`FRAPPE_OAUTH_CLIENT_ID` unset — local dev / tests),
    it falls back to an empty bearer so `tools.py` still runs against a stubbed
    Frappe; with auth on, a missing token is an error.
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        auth_on = bool(get_settings().frappe_oauth_client_id)
        access_token = get_access_token()
        tool_name = getattr(context.message, "name", None)

        if auth_on and tool_name in _WRITE_TOOL_NAMES:
            scopes = set(getattr(access_token, "scopes", None) or [])
            if WRITE_SCOPE not in scopes:
                raise ToolError(
                    "This connector token cannot make changes (missing the expenso:write scope)."
                )

        bearer = getattr(access_token, "token", None)
        if bearer is None:
            if auth_on:
                raise ToolError("Unauthenticated request.")
            bearer = ""

        client_token = bind_frappe_client(FrappeClient(bearer))
        entry_token = tools.bind_entry_method("connector")
        try:
            return await call_next(context)
        finally:
            tools.reset_entry_method(entry_token)
            reset_frappe_client(client_token)


def build_mcp_server() -> FastMCP:
    mcp = FastMCP("expenso-assistant", auth=build_auth_provider())
    mcp.add_middleware(MemberContextMiddleware())

    for fn in tools.READ_TOOLS:
        mcp.tool(fn)
    for fn in tools.WRITE_TOOLS:
        mcp.tool(_needs_write_confirmation(fn), name=fn.__name__)

    return mcp
