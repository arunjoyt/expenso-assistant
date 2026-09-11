"""Service configuration and the coupled model/pricing constant.

v1 ships on one OpenAI model; its id and its price live together here as a
hardcoded constant (ADR 0008). The OpenAI API returns token counts, not a
dollar figure — the service computes cost from this table and attaches it
explicitly to every Langfuse generation (P6-S5). A model swap is one reviewed
commit to `MODEL_PRICING`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class ModelRate:
    """USD per 1,000,000 tokens."""

    input: float
    cached_input: float
    output: float


MODEL_PRICING: dict[str, ModelRate] = {
    "gpt-4o-mini": ModelRate(input=0.15, cached_input=0.075, output=0.60),
    "gpt-4o": ModelRate(input=2.50, cached_input=1.25, output=10.00),
}


def cost_for(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> float | None:
    """Dollar cost of one generation, or None for an unpriced model.

    Cached input is billed at its discounted rate; reasoning tokens bill as
    output (ADR 0008's 2026-09-10 update).
    """
    rate = MODEL_PRICING.get(model)
    if rate is None:
        return None
    uncached_input = max(input_tokens - cached_tokens, 0)
    total = (
        uncached_input * rate.input
        + cached_tokens * rate.cached_input
        + (output_tokens + reasoning_tokens) * rate.output
    )
    return total / 1_000_000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI — one key for the whole product, admin-capped on the OpenAI dashboard.
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

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

    # Observability (P6-S5) — self-hosted Langfuse v2, loopback only in prod.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # LangGraph checkpointer + Langfuse share this Postgres (P6-S5).
    database_url: str = "postgresql://postgres:postgres@localhost:5432/assistant"

    # "Today" for the daily caps and the agent's date reasoning. The Family's
    # timezone context; one Family in v1.
    service_timezone: str = "Asia/Kolkata"

    # Per-run runaway guard (P6-S5). One agent->tools cycle is two graph
    # super-steps, so the recursion limit sits above 2 x run_max_tool_calls to
    # keep the tool-call cap the one that bites first; recursion is the backstop
    # for a pathological loop that never calls a tool.
    run_recursion_limit: int = 50
    run_max_tool_calls: int = 20
    run_wall_clock_seconds: int = 90

    # Per-Member daily token cap — input+output tokens across chat/receipt
    # turns, counted from a local Postgres total rather than Langfuse. No
    # fail-open: the counter lives in the checkpointer's own Postgres, already
    # a hard dependency for a turn to run at all (ADR 0008's 2026-09-11 update
    # — replaces the old turn-count `daily_chat_cap`).
    daily_token_cap: int = 500_000

    # How much of a thread's persisted history is resent to the model on each
    # turn — bounds the token count, not the message count, since tool-result
    # messages vary hugely in size (ADR 0008's 2026-09-11 update). Applied
    # transiently in `agent_node`; the checkpointed thread itself is never
    # trimmed, so `GET /history` and "Clear chat" are unaffected.
    chat_history_token_budget: int = 20_000

    # A secondary bound alongside the daily token cap: no single message can
    # consume an outsized share of one day's budget in one turn (ADR 0008's
    # 2026-09-11 update).
    max_chat_message_chars: int = 4_000

    # Confirm-card batch cap (P6-S7): the propose node shows at most this many
    # proposed writes per card and tells the model to continue with the rest.
    # This replaced the daily write cap — every in-app write is Member-confirmed,
    # so the useful bound is blast radius per confirmation, not a daily total
    # (ADR 0008's 2026-09-10 P6-S7 update).
    max_proposed_writes_per_turn: int = 25

    http_timeout_seconds: float = 30.0

    # A backstop against a misbehaving client, not a format negotiation (P7-S1):
    # the frontend always re-encodes to JPEG and downscales client-side, so a
    # well-behaved upload is far under this. Base64 chars, not decoded bytes.
    max_receipt_image_chars: int = 8_000_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
