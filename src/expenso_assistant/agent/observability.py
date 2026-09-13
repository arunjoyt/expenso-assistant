"""Langfuse tracing, the explicit-cost callback, and the daily token cap.

ADR 0008: Langfuse is the *only* record of an LLM call. Every turn opens one
trace tagged `user_id` / `metadata.feature` / `session_id` (no `family` — see the
2026-09-10 P6-S5 update), and the generation carries a cost this service
computes from `config.MODEL_PRICING` — never Langfuse's own model-price table.

The stock `langfuse.callback.CallbackHandler` is deliberately not used: it defers
cost to Langfuse's catalog, and it drags in the full `langchain` meta-package.

The daily cap (`within_daily_token_cap`) is the one exception to "Langfuse is
the only record": it reads a same-day token total from a small Postgres
counter, not from Langfuse (ADR 0008's 2026-09-11 update) — see
`PostgresTokenStore`'s docstring for why that doesn't reopen the rule above.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langfuse import Langfuse

from ..config import cost_for, get_settings

logger = logging.getLogger(__name__)

FEATURE_CHAT = "chat"
FEATURE_RECEIPT = "receipt"
# Proactive scheduled runs (P7-S2) — never counted against `daily_token_cap`
# (see `within_daily_token_cap` below): volume is inherently tiny (at most one
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


class PostgresTokenStore:
    """The daily per-Member token total backing `within_daily_token_cap`
    (ADR 0008's 2026-09-11 update, replacing the turn-count `daily_chat_cap`).
    A same-day, increment-only numeric aggregate with no per-call detail —
    not a record of an LLM call, so this does not reopen this module's
    "Langfuse is the *only* record" rule above. Lives as a table in the
    LangGraph checkpointer's own Postgres (`config.database_url`), not a new
    datastore."""

    def __init__(self, database_url: str):
        import psycopg

        self._conn = psycopg.connect(database_url, autocommit=True)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_daily_token_usage (
                user_id TEXT NOT NULL,
                usage_date DATE NOT NULL,
                tokens BIGINT NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, usage_date)
            )
            """
        )

    def get(self, user_id: str, today: date) -> int:
        row = self._conn.execute(
            "SELECT tokens FROM assistant_daily_token_usage WHERE user_id = %s AND usage_date = %s",
            (user_id, today),
        ).fetchone()
        return row[0] if row else 0

    def add(self, user_id: str, today: date, tokens: int) -> None:
        self._conn.execute(
            """
            INSERT INTO assistant_daily_token_usage (user_id, usage_date, tokens)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, usage_date)
            DO UPDATE SET tokens = assistant_daily_token_usage.tokens + EXCLUDED.tokens
            """,
            (user_id, today, tokens),
        )


_token_store: Any | None = None


def token_store() -> Any:
    global _token_store
    if _token_store is None:
        _token_store = PostgresTokenStore(get_settings().database_url)
    return _token_store


def reset_token_store() -> None:
    """Drop the cached store — tests and a settings reload call this."""
    global _token_store
    _token_store = None


def within_daily_token_cap(user_id: str) -> bool:
    """Whether this Member may start another chat or receipt turn today, by
    cumulative token total rather than turn count — a long message or a
    tool-heavy/vision turn no longer hides inside a flat per-turn allowance
    (ADR 0008's 2026-09-11 update). No fail-open branch: the counter's
    storage is the LangGraph checkpointer's own Postgres, already a hard
    dependency for the turn to run at all, so a failure here surfaces the
    same way a checkpointer failure downstream would — the caller (`session.
    stream_turn`) turns any exception into the turn's ordinary 'internal'
    error rather than silently permitting the turn."""
    cap = get_settings().daily_token_cap
    return token_store().get(user_id, _local_today()) < cap


def _local_midnight() -> datetime:
    now = datetime.now(ZoneInfo(get_settings().service_timezone))
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _local_today() -> date:
    return _local_midnight().date()


# --- per-turn trace + cost ------------------------------------------------


class TurnTrace:
    """One Langfuse trace for one chat turn, plus the callback that records each
    generation's cost explicitly."""

    def __init__(self, trace: Any, model: str, user_id: str):
        self._trace = trace
        self.callback = CostCallback(trace, model, user_id)

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
        return cls(trace, get_settings().openai_model, user_id)

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
    def __init__(self, trace: Any, model: str, user_id: str):
        self._trace = trace
        self._model = model
        self._user_id = user_id
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
        # Best-effort bookkeeping, unlike the gate in `within_daily_token_cap`:
        # the call already happened and the cost is already incurred, so a
        # failure here is logged and swallowed rather than failing the turn
        # after the fact (same trade-off as `_flush_trace`'s Langfuse flush).
        try:
            token_store().add(
                self._user_id, _local_today(), usage["input_tokens"] + usage["output_tokens"]
            )
        except Exception as exc:
            logger.warning("daily token counter update failed for %s: %s", self._user_id, exc)


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
    if not token_usage:
        # expenso-assistant#4: a live trace comparison found $0.00/empty usage on
        # generations that end in a tool call with no final text — unconfirmed
        # whether that's the actual trigger. Logged here (not raised) so a live
        # recurrence tells us, from `tool_calls`/`finish_reason`, whether that
        # hypothesis holds, without guessing further from static reading alone.
        logger.warning(
            "no usage metadata on LLM response (tool_calls=%s, finish_reason=%s, content=%r)",
            bool(getattr(message, "tool_calls", None)),
            (getattr(message, "response_metadata", None) or {}).get("finish_reason"),
            getattr(message, "content", None),
        )
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
