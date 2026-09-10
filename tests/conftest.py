import pytest

from expenso_assistant import frappe_client
from expenso_assistant.config import get_settings

FRAPPE_URL = "http://frappe.test"
API = "expenso.expenso.api"


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setenv("FRAPPE_URL", FRAPPE_URL)
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("SERVICE_TIMEZONE", "UTC")
    monkeypatch.setenv("RUN_WALL_CLOCK_SECONDS", "30")
    monkeypatch.delenv("FRAPPE_OAUTH_CLIENT_ID", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def spy_langfuse(monkeypatch):
    """Replace the Langfuse client with a recording spy. Returns a factory so a
    test can set the trace count / a fetch error before the first use."""
    from expenso_assistant.agent import observability

    from .fakes import SpyLangfuse

    def install(**kwargs) -> SpyLangfuse:
        spy = SpyLangfuse(**kwargs)
        monkeypatch.setattr(observability, "_client", spy)
        return spy

    observability.reset_langfuse_client()
    yield install
    observability.reset_langfuse_client()


@pytest.fixture
def member():
    from expenso_assistant.auth import AuthedMember, thread_id_for

    email = "priya@example.com"
    return AuthedMember(
        email=email, thread_id=thread_id_for(email), scopes=("expenso:read",), token="member-bearer"
    )


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
