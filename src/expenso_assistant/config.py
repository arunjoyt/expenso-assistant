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

    # Observability (P6-S5) — self-hosted Langfuse v2, loopback only in prod.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # LangGraph checkpointer + Langfuse share this Postgres (P6-S5).
    database_url: str = "postgresql://postgres:postgres@localhost:5432/assistant"

    # Per-run runaway guard (P6-S5).
    run_recursion_limit: int = 25
    run_max_tool_calls: int = 20
    run_wall_clock_seconds: int = 90

    # Per-Member daily caps, counted from Langfuse, fail-open (P6-S5).
    daily_chat_cap: int = 50
    daily_receipt_cap: int = 30
    daily_write_cap: int = 100

    http_timeout_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
