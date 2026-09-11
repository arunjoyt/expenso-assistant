"""Langfuse tracing, the explicit-cost callback, and the daily-cap query.

ADR 0008: Langfuse is the *only* record of an LLM call. Every turn opens one
trace tagged `user_id` / `metadata.feature` / `session_id` (no `family` — see the
2026-09-10 P6-S5 update), and the generation carries a cost this service
computes from `config.MODEL_PRICING` — never Langfuse's own model-price table.

The stock `langfuse.callback.CallbackHandler` is deliberately not used: it defers
cost to Langfuse's catalog, and it drags in the full `langchain` meta-package.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langfuse import Langfuse

from ..config import cost_for, get_settings

logger = logging.getLogger(__name__)

FEATURE_CHAT = "chat"
FEATURE_RECEIPT = "receipt"
# Proactive scheduled runs (P7-S2) — never counted against `daily_chat_cap`
# (see `within_daily_chat_cap` below): volume is inherently tiny (at most one
# monthly + one weekly LLM call per Member, and the budget-drift pre-check
# skips the LLM call most weeks), so a separate cap isn't worth the config
# surface.
FEATURE_INSIGHTS = "insights"

_client: Langfuse | None = None


def langfuse_client() -> Langfuse:
    global _client
    if _client is None:
        s = get_settings()
        _client = Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
        )
    return _client


def reset_langfuse_client() -> None:
    """Drop the cached client — tests and a settings reload call this."""
    global _client
    _client = None


# --- daily cap -------------------------------------------------------------


def within_daily_chat_cap(user_id: str) -> bool:
    """Whether this Member may start another chat or receipt turn today — the
    two share one cap (P7-S1 grill: receipt volume is low and vision cost is
    already bounded per-run, so a second config knob isn't worth it). Fails
    **open** (ADR 0008) — a Langfuse outage never blocks a turn; the per-run
    caps still bound it."""
    cap = get_settings().daily_chat_cap
    try:
        counted = sum(
            _count_today(user_id, feature, cap) for feature in (FEATURE_CHAT, FEATURE_RECEIPT)
        )
        return counted < cap
    except Exception as exc:
        logger.warning("daily-cap check failed open for %s: %s", user_id, exc)
        return True


def _count_today(user_id: str, feature: str, limit: int) -> int:
    """Traces opened for this Member + feature since local midnight. Capped at
    `limit` — the caller only needs the "< cap?" answer. Must run *before* this
    turn's own trace opens, or it counts itself."""
    start = _local_midnight()
    result = langfuse_client().fetch_traces(
        user_id=user_id,
        tags=[f"feature:{feature}"],
        from_timestamp=start,
        to_timestamp=datetime.now(start.tzinfo),
        limit=limit,
    )
    return len(result.data)


def _local_midnight() -> datetime:
    now = datetime.now(ZoneInfo(get_settings().service_timezone))
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


# --- per-turn trace + cost ------------------------------------------------


class TurnTrace:
    """One Langfuse trace for one chat turn, plus the callback that records each
    generation's cost explicitly."""

    def __init__(self, trace: Any, model: str):
        self._trace = trace
        self.callback = CostCallback(trace, model)

    @classmethod
    def start(
        cls, *, user_id: str, session_id: str, user_input: str, feature: str = FEATURE_CHAT
    ) -> TurnTrace:
        trace = langfuse_client().trace(
            name="chat-turn",
            user_id=user_id,
            session_id=session_id,
            input=user_input,
            tags=[f"feature:{feature}"],
            metadata={"feature": feature},
        )
        return cls(trace, get_settings().openai_model)

    @property
    def total_cost(self) -> float:
        return self.callback.total_cost

    def finish(self, *, output: str) -> None:
        self._trace.update(output=output)

    def fail(self, code: str) -> None:
        self._trace.update(level="ERROR", status_message=code)

    def score(self, name: str, value: float) -> None:
        self._trace.score(name=name, value=value)


class CostCallback(BaseCallbackHandler):
    def __init__(self, trace: Any, model: str):
        self._trace = trace
        self._model = model
        self.total_cost = 0.0

    def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        usage = _usage_from_result(response)
        cost = cost_for(self._model, **usage) or 0.0
        self.total_cost += cost
        self._trace.generation(
            name="chat-completion",
            model=self._model,
            output=_text_from_result(response),
            usage_details={
                "input": usage["input_tokens"],
                "output": usage["output_tokens"],
                "cache_read_input_tokens": usage["cached_tokens"],
            },
            cost_details={"total": cost},
        )


def _usage_from_result(response: LLMResult) -> dict[str, int]:
    generation = response.generations[0][0]
    message = getattr(generation, "message", None)
    meta = getattr(message, "usage_metadata", None) or {}
    if meta:
        return {
            "input_tokens": meta.get("input_tokens", 0),
            "output_tokens": meta.get("output_tokens", 0),
            "cached_tokens": (meta.get("input_token_details") or {}).get("cache_read", 0),
            "reasoning_tokens": (meta.get("output_token_details") or {}).get("reasoning", 0),
        }
    token_usage = (response.llm_output or {}).get("token_usage", {})
    return {
        "input_tokens": token_usage.get("prompt_tokens", 0),
        "output_tokens": token_usage.get("completion_tokens", 0),
        "cached_tokens": (token_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "reasoning_tokens": (token_usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens", 0
        ),
    }


def _text_from_result(response: LLMResult) -> str:
    generation = response.generations[0][0]
    return getattr(generation, "text", "") or ""
