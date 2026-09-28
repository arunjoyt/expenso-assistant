"""Thin async Frappe REST client — bearer passthrough, nothing clever.

Every tool call reaches Frappe *as the Member*: the caller's OAuth bearer is
forwarded verbatim, so `validate_oauth()` → `frappe.set_user()` and every
`permission_query_conditions` / `has_permission` hook run exactly as they do
for a first-party request. Family-scoping is therefore Frappe's job, not this
client's (ADR 0008).
"""

from __future__ import annotations

import contextvars
import json
import re

import httpx

from .config import get_settings

_current_client: contextvars.ContextVar[FrappeClient] = contextvars.ContextVar(
    "current_frappe_client"
)

_http: httpx.AsyncClient | None = None


def _shared_http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(timeout=get_settings().http_timeout_seconds)
    return _http


async def aclose_http() -> None:
    global _http
    if _http is not None and not _http.is_closed:
        await _http.aclose()
    _http = None


class FrappeError(RuntimeError):
    """A Frappe error response. `detail` is the raw body for logs and may hold
    a traceback; `user_message` is only what Frappe would show the Member."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.detail = body[:500]
        self.exc_type, self.user_message = _parse_error(status_code, body)
        super().__init__(f"Frappe REST {status_code}: {self.detail}")


class FrappeClient:
    def __init__(self, bearer_token: str, *, base_url: str | None = None):
        self._bearer = bearer_token
        self._base_url = (base_url or get_settings().frappe_url).rstrip("/")

    async def call(self, method: str, *, write: bool = False, **params):
        """Invoke a whitelisted method. Reads GET, writes POST. Returns `message`."""
        url = f"{self._base_url}/api/method/{method}"
        headers = {"Authorization": f"Bearer {self._bearer}", "Accept": "application/json"}
        payload = {key: value for key, value in params.items() if value is not None}
        http = _shared_http()

        if write:
            response = await http.post(url, headers=headers, json=payload)
        else:
            response = await http.get(url, headers=headers, params=payload)

        if response.status_code >= 400:
            raise FrappeError(response.status_code, response.text)
        return response.json().get("message")


def bind_frappe_client(client: FrappeClient) -> contextvars.Token:
    return _current_client.set(client)


def reset_frappe_client(token: contextvars.Token) -> None:
    _current_client.reset(token)


def current_frappe_client() -> FrappeClient:
    try:
        return _current_client.get()
    except LookupError as exc:
        raise RuntimeError("No FrappeClient bound to the current context") from exc


def _parse_error(status_code: int, body: str) -> tuple[str | None, str]:
    """(exc_type, user_message) from a Frappe v1 error body. The Member-facing
    text is `_server_messages`: a JSON list of JSON-encoded message dicts.
    `exc` and `exception` carry the traceback and are never read here."""
    try:
        data = json.loads(body)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return None, f"HTTP {status_code}"
    messages = [text for text in _server_messages(data) if text]
    exc_type = data.get("exc_type")
    return exc_type, " ".join(messages) or exc_type or f"HTTP {status_code}"


def _server_messages(data: dict) -> list[str]:
    try:
        entries = [json.loads(entry) for entry in json.loads(data.get("_server_messages") or "[]")]
    except (TypeError, ValueError):
        return []
    return [
        _HTML_TAG.sub("", str(entry.get("message", ""))).strip()
        for entry in entries
        if isinstance(entry, dict)
    ]


_HTML_TAG = re.compile(r"<[^>]+>")
