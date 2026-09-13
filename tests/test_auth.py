import httpx
import respx

from expenso_assistant.auth import FrappeTokenVerifier, build_auth_provider

from .conftest import FRAPPE_URL

INTROSPECT = f"{FRAPPE_URL}/api/method/frappe.integrations.oauth2.introspect_token"


def test_no_auth_provider_without_an_upstream_client():
    assert build_auth_provider() is None


def test_oauth_proxy_built_when_configured(monkeypatch):
    monkeypatch.setenv("FRAPPE_OAUTH_CLIENT_ID", "abc")
    monkeypatch.setenv("FRAPPE_OAUTH_CLIENT_SECRET", "shh")
    from expenso_assistant.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    provider = build_auth_provider()
    assert provider is not None
    assert type(provider).__name__ == "OAuthProxy"
    # Live prod bug: base_url must match where the OAuth routes actually
    # resolve. main.py mounts the whole MCP app (protocol + OAuth routes) at
    # the ASGI root, with only the protocol endpoint pushed to `/mcp`
    # internally — so base_url is the bare public root, not `.../mcp`. (A
    # prior fix attempt appended "/mcp" here instead of fixing the mount;
    # that made FastMCP apply RFC 8414's path-suffix convention to its own
    # discovery-document address, a different 404.)
    assert str(provider.base_url).rstrip("/") == settings.public_base_url.rstrip("/")
    # Live prod bug: FastMCP's OAuthProxy defaults to "client_secret_basic"
    # (an Authorization: Basic header) for the upstream token exchange, but
    # Frappe's `get_token` only ever reads credentials from the POST body
    # (see DEPLOYMENT.md's "prefer PKCE" caveat) — the basic header is
    # silently ignored and the exchange fails, surfaced to a connecting
    # client as an opaque "Authorization failed".
    assert provider._token_endpoint_auth_method == "client_secret_post"


@respx.mock
async def test_verifier_accepts_an_active_token_and_reads_its_scopes():
    respx.post(INTROSPECT).mock(
        return_value=httpx.Response(
            200,
            json={
                "active": True,
                "scope": "all openid expenso:read",
                "client_id": "c1",
                "exp": 9999999999,
            },
        )
    )

    token = await FrappeTokenVerifier(frappe_url=FRAPPE_URL).verify_token("good")

    assert token is not None
    assert token.token == "good"
    assert "expenso:read" in token.scopes


@respx.mock
async def test_verifier_rejects_an_inactive_token():
    respx.post(INTROSPECT).mock(return_value=httpx.Response(200, json={"active": False}))

    assert await FrappeTokenVerifier(frappe_url=FRAPPE_URL).verify_token("stale") is None


@respx.mock
async def test_verifier_returns_none_when_frappe_is_unreachable():
    respx.post(INTROSPECT).mock(side_effect=httpx.ConnectError("down"))

    assert await FrappeTokenVerifier(frappe_url=FRAPPE_URL).verify_token("x") is None
