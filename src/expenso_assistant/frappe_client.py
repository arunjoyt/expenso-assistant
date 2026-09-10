"""Thin async Frappe REST client — bearer passthrough, nothing clever.

Every tool call reaches Frappe *as the Member*: the caller's OAuth bearer is
forwarded verbatim, so `validate_oauth()` → `frappe.set_user()` and every
`permission_query_conditions` / `has_permission` hook run exactly as they do
for a first-party request. Family-scoping is therefore Frappe's job, not this
client's (ADR 0008).
"""

from __future__ import annotations

import contextvars

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
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"Frappe REST {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


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
            raise FrappeError(response.status_code, response.text[:500])
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
