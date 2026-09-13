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
    provider = build_auth_provider()
    assert provider is not None
    assert type(provider).__name__ == "OAuthProxy"
    # Live prod bug: the MCP app is mounted at `/mcp` (main.py), so every
    # self-advertised OAuth URL (discovery metadata, WWW-Authenticate) must
    # carry that prefix or a connecting client is sent to a 404 `/authorize`.
    assert str(provider.base_url).rstrip("/").endswith("/mcp")


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
