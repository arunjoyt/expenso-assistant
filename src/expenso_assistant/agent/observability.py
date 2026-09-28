"""LangSmith tracing, receipt-accuracy feedback, and the daily token cap.

ADR 0011 (supersedes ADR 0008's self-hosted Langfuse): every turn is one
LangSmith trace, recorded by LangChain's stock `LangChainTracer`. The turn's
root run carries `user_id` / `session_id` / `feature` as metadata and a
`feature:<x>` tag; the tracer nests every model call, tool call, middleware
step and confirm-card pause under it, and records errors and outputs itself.
LangSmith prices each model call from its own model table.

Traces go to hosted LangSmith (US region by default), so family financial data
leaves our infrastructure. Receipt images never do: `mask_images` replaces
every image data URI in a run's inputs before upload. Amounts, categories,
notes and member emails are sent as-is (ADR 0011).

Tracing is off when `LANGSMITH_API_KEY` is empty; the key alone turns it on.
Leave `LANGSMITH_TRACING` unset: a turn already carries this tracer, so
LangChain adds none of its own, but any LangChain call made outside a turn
would be traced by LangChain's default client, which does not mask images.

The daily cap (`within_daily_token_cap`) reads a same-day token total from a
small Postgres counter, not from the tracing backend (ADR 0008's 2026-09-11
update).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.tracers import LangChainTracer
from langsmith import Client

from ..config import get_settings

logger = logging.getLogger(__name__)

FEATURE_CHAT = "chat"
FEATURE_RECEIPT = "receipt"
# Proactive scheduled runs (P7-S2) — never counted against `daily_token_cap`
# (see `within_daily_token_cap` below): volume is inherently tiny (at most one
# monthly + one weekly LLM call per Member, and the budget-drift pre-check
# skips the LLM call most weeks), so a separate cap isn't worth the config
# surface.
FEATURE_INSIGHTS = "insights"

IMAGE_PLACEHOLDER = "[receipt image removed]"

_client: Client | None = None


def langsmith_client() -> Client | None:
    """The shared LangSmith client, or None when tracing is off."""
    global _client
    settings = get_settings()
    if _client is None and settings.langsmith_api_key:
        _client = Client(
            api_url=settings.langsmith_endpoint,
            api_key=settings.langsmith_api_key,
            hide_inputs=mask_images,
        )
    return _client


def reset_langsmith_client() -> None:
    """Drop the cached client — tests and a settings reload call this."""
    global _client
    _client = None


def tracers() -> list[BaseCallbackHandler]:
    """The stock LangSmith tracer for one run, or none when tracing is off."""
    client = langsmith_client()
    if client is None:
        return []
    return [LangChainTracer(client=client, project_name=get_settings().langsmith_project)]


def flush_traces() -> None:
    """Send what the client still holds. The app calls this at shutdown."""
    client = langsmith_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.warning("langsmith flush failed: %s", exc)


def mask_images(inputs: dict) -> dict:
    """The client's `hide_inputs` hook: a receipt reaches the model as a
    `data:image/...;base64,` URI (P7-S1), and never reaches LangSmith. Model
    runs record messages as dicts; middleware runs record the live message
    objects of the model request, so both are walked."""
    return _mask(inputs)


def _mask(value: Any) -> Any:
    if isinstance(value, str):
        return IMAGE_PLACEHOLDER if value.startswith("data:image/") else value
    if isinstance(value, BaseMessage):
        return value.model_copy(update={"content": _mask(value.content)})
    if isinstance(value, dict):
        return {key: _mask(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_mask(item) for item in value]
    return value


# --- daily cap -------------------------------------------------------------


class PostgresTokenStore:
    """The daily per-Member token total backing `within_daily_token_cap`
    (ADR 0008's 2026-09-11 update, replacing the turn-count `daily_chat_cap`).
    A same-day, increment-only numeric aggregate with no per-call detail —
    not a record of an LLM call; the trace stays that record. Lives as a table in the
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


# Receipt extraction accuracy (ADR 0003). A proposed `None` is excluded — there
# was no claim to grade.
_SCORED_FIELDS = ("amount", "date", "category", "notes")


def score_receipt_accuracy(card: dict, decision: dict, trace: TurnTrace) -> None:
    """The confirm-time correction signal (ADR 0003): per field, did what the
    vision call proposed survive the member's review? Posted on the resume
    leg's own trace (P7-S1), where the comparison happens. Reads the stock HITL
    card and decisions (ADR 0010), which line up one-to-one; rejected rows are
    not scored."""
    requests = card.get("action_requests") or []
    for request, choice in zip(requests, decision.get("decisions") or [], strict=False):
        if request.get("name") != "create_expense" or choice.get("type") == "reject":
            continue
        proposed = request.get("args") or {}
        confirmed = (choice.get("edited_action") or {}).get("args") or proposed
        for field in _SCORED_FIELDS:
            if proposed.get(field) is None:
                continue
            match = str(proposed[field]) == str(confirmed.get(field))
            trace.score(f"receipt_accuracy_{field}", 1.0 if match else 0.0)


def _local_midnight() -> datetime:
    now = datetime.now(ZoneInfo(get_settings().service_timezone))
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _local_today() -> date:
    return _local_midnight().date()


# --- per-turn trace + token count ------------------------------------------


class TurnTrace:
    """One LangSmith trace for one leg of a turn: `turn` for the message,
    `resume` for the confirm-card decision. `apply` puts the root run's id,
    name, tags and metadata on the run config, next to the tracer and the
    daily token counter."""

    def __init__(
        self, *, user_id: str, session_id: str, feature: str = FEATURE_CHAT, leg: str = "turn"
    ):
        self.run_id: UUID = uuid4()
        self._user_id = user_id
        self._session_id = session_id
        self._feature = feature
        self._leg = leg

    def apply(self, config: dict) -> dict:
        config.update(
            run_id=self.run_id,
            run_name=f"{self._feature}-{self._leg}",
            tags=[f"feature:{self._feature}"],
            # `session_id` groups a Member's turns into one LangSmith thread.
            metadata={
                "feature": self._feature,
                "user_id": self._user_id,
                "session_id": self._session_id,
            },
            callbacks=[*tracers(), TokenCounter(self._user_id)],
        )
        return config

    def score(self, key: str, value: float) -> None:
        """Feedback on this trace. `trace_id` lets the client batch it with the
        runs, so it may be sent before the run itself has ended."""
        client = langsmith_client()
        if client is None:
            return
        try:
            client.create_feedback(trace_id=self.run_id, key=key, score=value)
        except Exception as exc:
            logger.warning("langsmith feedback %s failed: %s", key, exc)


class TokenCounter(BaseCallbackHandler):
    """Adds each model call's tokens to the Member's daily total."""

    def __init__(self, user_id: str):
        self._user_id = user_id

    def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        usage = _usage_from_result(response)
        # Best-effort bookkeeping, unlike the gate in `within_daily_token_cap`:
        # the call already happened and the cost is already incurred, so a
        # failure here is logged and swallowed rather than failing the turn
        # after the fact.
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
    token_usage = (response.llm_output or {}).get("token_usage", {})

    usage = {
        "input_tokens": meta.get("input_tokens", 0),
        "output_tokens": meta.get("output_tokens", 0),
    }
    if not usage["input_tokens"] and not usage["output_tokens"]:
        usage = {
            "input_tokens": token_usage.get("prompt_tokens", 0),
            "output_tokens": token_usage.get("completion_tokens", 0),
        }
    if not usage["input_tokens"] and not usage["output_tokens"]:
        # expenso-assistant#4: a live trace comparison found empty usage on
        # generations that end in a tool call with no final text. The original
        # version of this warning only fired when `usage_metadata` was entirely
        # absent — a live recurrence showed it comes back as a non-empty dict
        # with all fields zeroed instead, which skipped that check silently.
        # Logging both raw sources now (not just the zeroed one) to see which
        # side, if either, actually carries real numbers.
        logger.warning(
            "zero usage on LLM response (tool_calls=%s, finish_reason=%s, content=%r, "
            "usage_metadata=%r, token_usage=%r, response_metadata=%r)",
            bool(getattr(message, "tool_calls", None)),
            (getattr(message, "response_metadata", None) or {}).get("finish_reason"),
            getattr(message, "content", None),
            meta,
            token_usage,
            getattr(message, "response_metadata", None),
        )
    return usage
