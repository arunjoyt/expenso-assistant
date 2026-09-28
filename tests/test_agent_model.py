"""The one place OpenAI is named (ADR 0008)."""

from __future__ import annotations

from expenso_assistant.agent.model import build_model
from expenso_assistant.config import get_settings


def test_streamed_calls_report_usage_behind_a_custom_base_url(monkeypatch):
    """ChatOpenAI drops streamed usage by itself when OPENAI_BASE_URL is set,
    which zeroes the daily token cap (expenso-assistant#4)."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm-proxy.example/v1")

    model = build_model(get_settings())

    assert model.stream_usage is True
