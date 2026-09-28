"""Service configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI — one key for the whole product, admin-capped on the OpenAI dashboard.
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Tried by the stock `ModelFallbackMiddleware` when `openai_model` still
    # fails after the SDK's own retries (ADR 0010). Empty turns fallback off.
    # Its tokens count against the daily cap like any other.
    openai_fallback_model: str = ""

    # Frappe is the OAuth authorization server and the system of record.
    frappe_url: str = "http://localhost:8000"
    frappe_oauth_client_id: str = ""
    frappe_oauth_client_secret: str = ""

    # This service's own externally reachable base URL (OAuth proxy callbacks).
    public_base_url: str = "http://localhost:8080"

    # The FastMCP external-connector adapter. A pure adapter — off has zero
    # effect on the in-app Assistant (P6-S5+).
    mcp_enabled: bool = True

    # Origins allowed to call the service from a browser (the in-app Assistant
    # tab, P6-S6). Comma-separated; the Frappe app's public origin(s). Empty in
    # dev where the frontend is same-origin-proxied or not browser-driven.
    allowed_cors_origins: str = ""

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_cors_origins.split(",") if origin.strip()]

    # LangGraph checkpointer (P6-S5) and the daily token counter.
    database_url: str = "postgresql://postgres:postgres@localhost:5432/assistant"

    # "Today" for the daily caps and the agent's date reasoning. The Family's
    # timezone context; one Family in v1.
    service_timezone: str = "Asia/Kolkata"

    # Per-run runaway guard (P6-S5). `run_max_tool_calls` feeds the stock
    # `ToolCallLimitMiddleware` (ADR 0010): it counts single tool calls, and
    # resets when `/resume` starts a new run. One `create_agent` cycle is up
    # to four graph super-steps (model, two `after_model` hooks, tools), so the
    # recursion limit sits above 4 x run_max_tool_calls to keep the tool-call
    # cap the one that bites first; recursion is the backstop for a
    # pathological loop that never calls a tool.
    run_recursion_limit: int = 100
    run_max_tool_calls: int = 20
    run_wall_clock_seconds: int = 90

    # Per-Member daily token cap — input+output tokens across chat/receipt
    # turns, counted from a local Postgres total rather than the traces. No
    # fail-open: the counter lives in the checkpointer's own Postgres, already
    # a hard dependency for a turn to run at all (ADR 0008's 2026-09-11 update
    # — replaces the old turn-count `daily_chat_cap`).
    daily_token_cap: int = 500_000

    # Above this many (approximate) tokens, the stock `ContextEditingMiddleware`
    # clears older tool outputs from what the model sees (ADR 0010) — tool
    # results are what vary hugely in size. Transient: the checkpointed thread
    # is never edited, so `GET /history` and "Clear chat" are unaffected.
    chat_history_token_budget: int = 20_000

    # A secondary bound alongside the daily token cap: no single message can
    # consume an outsized share of one day's budget in one turn (ADR 0008's
    # 2026-09-11 update).
    max_chat_message_chars: int = 4_000

    http_timeout_seconds: float = 30.0

    # A backstop against a misbehaving client, not a format negotiation (P7-S1):
    # the frontend always re-encodes to JPEG and downscales client-side, so a
    # well-behaved upload is far under this. Base64 chars, not decoded bytes.
    max_receipt_image_chars: int = 8_000_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
