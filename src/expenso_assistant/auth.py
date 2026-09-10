"""Resource-server auth for the FastMCP external adapter.

Frappe stays the OAuth **authorization server**; this service is a **resource
server**. External connectors (ChatGPT/Claude) run the standard Authorization
Code + PKCE flow against Frappe. Frappe has Dynamic Client Registration turned
off and issues opaque (non-JWT) bearer tokens, so:

- a FastMCP `OAuthProxy` fronts Frappe's `/authorize` + `/token` with one
  fixed, admin-registered `OAuth Client`, presenting DCR to MCP clients that
  expect it while the upstream call carries the fixed `client_id` and the
  client's PKCE `code_verifier` (never `Authorization: Basic` — Frappe ignores
  it there, see DEPLOYMENT.md).
- `FrappeTokenVerifier` validates an incoming bearer with Frappe's RFC 7662
  introspection endpoint and reads its granted scopes.

`build_auth_provider()` returns `None` when no upstream client is configured
(local dev / tests) — the FastMCP server then runs unauthenticated in-process.
"""

from __future__ import annotations

import httpx
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.auth.oauth_proxy import OAuthProxy

from .config import get_settings

_INTROSPECT = "/api/method/frappe.integrations.oauth2.introspect_token"
_AUTHORIZE = "/api/method/frappe.integrations.oauth2.authorize"
_TOKEN = "/api/method/frappe.integrations.oauth2.get_token"


class FrappeTokenVerifier(TokenVerifier):
    """Validate an opaque Frappe bearer via RFC 7662 introspection."""

    def __init__(self, *, frappe_url: str | None = None):
        super().__init__()
        self._url = (frappe_url or get_settings().frappe_url).rstrip("/")

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            async with httpx.AsyncClient(timeout=get_settings().http_timeout_seconds) as http:
                response = await http.post(
                    f"{self._url}{_INTROSPECT}",
                    data={"token": token, "token_type_hint": "access_token"},
                )
        except httpx.HTTPError:
            return None

        if response.status_code != 200:
            return None
        body = response.json()
        if not body.get("active"):
            return None

        scopes = (body.get("scope") or "").split()
        return AccessToken(
            token=token,
            client_id=str(body.get("client_id") or "frappe"),
            scopes=scopes,
            expires_at=body.get("exp"),
        )


def build_auth_provider():
    """The FastMCP `auth=` provider, or None when unconfigured (dev/tests)."""
    settings = get_settings()
    if not settings.frappe_oauth_client_id:
        return None

    frappe_url = settings.frappe_url.rstrip("/")
    return OAuthProxy(
        upstream_authorization_endpoint=f"{frappe_url}{_AUTHORIZE}",
        upstream_token_endpoint=f"{frappe_url}{_TOKEN}",
        upstream_client_id=settings.frappe_oauth_client_id,
        upstream_client_secret=settings.frappe_oauth_client_secret,
        token_verifier=FrappeTokenVerifier(frappe_url=frappe_url),
        base_url=settings.public_base_url,
    )
