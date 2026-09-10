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

The **in-app Assistant** endpoints use the same introspection, then resolve the
Member: introspection alone can't be trusted for identity (Frappe only returns
`sub` when a `User Social Login` row exists), so after the token checks out
`resolve_member()` calls the stock `frappe.auth.get_logged_user` as the Member.
The chat thread is 1:1 with the Member — `thread_id_for()` derives an opaque id
from the email, and no client ever supplies one (P6-S5 grill, Q2/Q3).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import httpx
from fastapi import Depends, HTTPException, Request
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.auth.oauth_proxy import OAuthProxy

from .config import get_settings

_INTROSPECT = "/api/method/frappe.integrations.oauth2.introspect_token"
_AUTHORIZE = "/api/method/frappe.integrations.oauth2.authorize"
_TOKEN = "/api/method/frappe.integrations.oauth2.get_token"
_LOGGED_USER = "/api/method/frappe.auth.get_logged_user"

READ_SCOPE = "expenso:read"


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


@dataclass(frozen=True)
class AuthedMember:
    email: str
    thread_id: str
    scopes: tuple[str, ...]
    token: str


def thread_id_for(email: str) -> str:
    """One opaque, stable thread id per Member. Opaque so it is not PII in
    Langfuse / Postgres keys; derived, so the client cannot name someone
    else's thread."""
    return "member:" + hashlib.sha256(email.strip().lower().encode()).hexdigest()


async def resolve_member(token: str, *, frappe_url: str | None = None) -> AuthedMember | None:
    """Introspect the bearer, require `expenso:read`, and resolve the Member via
    `get_logged_user`. `None` for any failure — the endpoint turns that into 401."""
    access = await FrappeTokenVerifier(frappe_url=frappe_url).verify_token(token)
    if access is None or READ_SCOPE not in (access.scopes or []):
        return None

    url = (frappe_url or get_settings().frappe_url).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=get_settings().http_timeout_seconds) as http:
            response = await http.get(
                f"{url}{_LOGGED_USER}", headers={"Authorization": f"Bearer {token}"}
            )
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    email = response.json().get("message")
    if not email or email == "Guest":
        return None

    return AuthedMember(
        email=email,
        thread_id=thread_id_for(email),
        scopes=tuple(access.scopes or []),
        token=token,
    )


async def require_member(request: Request) -> AuthedMember:
    """FastAPI dependency for the in-app Assistant endpoints."""
    header = request.headers.get("Authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    member = await resolve_member(token) if token else None
    if member is None:
        raise HTTPException(status_code=401, detail="Invalid or expired Assistant token")
    return member


MemberDep = Depends(require_member)


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
