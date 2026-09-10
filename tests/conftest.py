import pytest

from expenso_assistant import frappe_client
from expenso_assistant.config import get_settings

FRAPPE_URL = "http://frappe.test"
API = "expenso.expenso.api"


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setenv("FRAPPE_URL", FRAPPE_URL)
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.delenv("FRAPPE_OAUTH_CLIENT_ID", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
async def _reset_http():
    yield
    await frappe_client.aclose_http()


@pytest.fixture
def bound_client():
    """Bind a FrappeClient for the calling Member for the duration of a test."""
    token = frappe_client.bind_frappe_client(frappe_client.FrappeClient("member-bearer"))
    try:
        yield
    finally:
        frappe_client.reset_frappe_client(token)


def method_url(method: str) -> str:
    return f"{FRAPPE_URL}/api/method/{method}"
