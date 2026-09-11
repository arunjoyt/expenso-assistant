"""Proactive (scheduler-triggered) runs — no SSE, no live client on the other
end. Frappe's `hooks.py` `scheduler_events` mints a per-Member read-scoped
token and POSTs `/run/proactive {"job": ...}`; `api/main.py` hands the request
off to `run_proactive_job` as a background task and returns `202` immediately.

Both jobs bind `READ_TOOLS` only (`build_graph(..., tools=tool_defs.READ_TOOLS)`
in `api/main.py`) — a proactive run cannot produce a write tool-call, so it
never proposes anything, only ever an Insight message. There is no dedup
marker: `run_budget_drift` re-warns every week a Category is still over its
threshold. See ADR 0008's 2026-09-11 P7-S2 update for the full reasoning.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from .. import tools as tool_defs
from ..auth import AuthedMember
from ..config import Settings
from ..frappe_client import FrappeClient, bind_frappe_client, reset_frappe_client
from .observability import FEATURE_INSIGHTS, TurnTrace
from .session import discard_pending, pending_card

logger = logging.getLogger(__name__)

JOB_MONTHLY_SUMMARY = "monthly_summary"
JOB_BUDGET_DRIFT = "budget_drift"

_DRIFT_STATUSES = ("Warning", "Exceeded")

_MONTHS = (
    "January February March April May June July August September October November December".split()
)


async def run_proactive_job(*, job: str, member: AuthedMember, graph, settings: Settings) -> None:
    """The background task `POST /run/proactive` kicks off. Any failure is
    logged and swallowed here — a proactive job has nothing to report an error
    back to, and it will simply try again on the next scheduled run."""
    try:
        if job == JOB_MONTHLY_SUMMARY:
            await _run_monthly_summary(member=member, graph=graph, settings=settings)
        elif job == JOB_BUDGET_DRIFT:
            await _run_budget_drift(member=member, graph=graph, settings=settings)
        else:
            logger.warning("unknown proactive job %r for %s", job, member.email)
    except Exception:
        logger.exception("proactive job %r failed for %s", job, member.email)


async def _run_monthly_summary(*, member: AuthedMember, graph, settings: Settings) -> None:
    """Always runs — even a quiet month is worth a summary."""
    today = _today(settings)
    prev_month, prev_year = _previous_month(today)
    span = f"{_MONTHS[prev_month - 1]} {prev_year}"
    instruction = (
        f"Write the family's monthly spending summary for {span}, the month that just "
        "ended. Read whatever you need (expenses, income, analytics, budgets) and write "
        "a short, friendly summary of how the month went."
    )
    await _post_insight(member=member, graph=graph, settings=settings, instruction=instruction)


async def _run_budget_drift(*, member: AuthedMember, graph, settings: Settings) -> None:
    """Gated by a deterministic pre-check — no LLM call at all when nothing is
    currently over budget (P7-S2 grill: not dedup, just not spamming "you're
    fine" every week)."""
    today = _today(settings)
    crossed = await _categories_over_budget(member, today.month, today.year)
    if not crossed:
        return

    instruction = (
        f"These categories are currently over or near their budget for "
        f"{_MONTHS[today.month - 1]} {today.year}: {', '.join(crossed)}. Read whatever you "
        "need to confirm the details and write a short, friendly Insight flagging this to "
        "the member."
    )
    await _post_insight(member=member, graph=graph, settings=settings, instruction=instruction)


async def _categories_over_budget(member: AuthedMember, month: int, year: int) -> list[str]:
    """Category names currently `Warning`/`Exceeded` for the month, reusing
    `get_analytics`'s `budget_status` field (`expenso_budget.py::compute_budget_status`
    on the Frappe side) rather than reimplementing the threshold here."""
    token = bind_frappe_client(FrappeClient(member.token))
    try:
        analytics = await tool_defs.get_analytics(month, year)
    finally:
        reset_frappe_client(token)
    return [
        category["name"]
        for category in analytics.get("categories") or []
        if category.get("budget_status") in _DRIFT_STATUSES
    ]


async def _post_insight(
    *, member: AuthedMember, graph, settings: Settings, instruction: str
) -> None:
    config = _proactive_config(member, settings, instruction=instruction)

    # A proactive run touching this thread discards any stale unconfirmed
    # proposal, same rule `stream_turn` already applies to a new live message
    # (P7-S2 grill: "the thread is linear").
    if await pending_card(graph, config) is not None:
        await discard_pending(graph, config)

    trace = TurnTrace.start(
        user_id=member.email,
        session_id=member.thread_id,
        user_input=instruction,
        feature=FEATURE_INSIGHTS,
    )
    config["callbacks"] = [trace.callback]

    client_token = bind_frappe_client(FrappeClient(member.token))
    entry_token = tool_defs.bind_entry_method("assistant")
    try:
        inputs = {"messages": [], "tool_call_count": 0, "entry_method": "assistant"}
        result = await asyncio.wait_for(
            graph.ainvoke(inputs, config), timeout=settings.run_wall_clock_seconds
        )
        trace.finish(output=_final_text(result))
    except Exception:
        trace.fail("internal")
        raise
    finally:
        tool_defs.reset_entry_method(entry_token)
        reset_frappe_client(client_token)
        _flush_trace()


def _proactive_config(member: AuthedMember, settings: Settings, *, instruction: str) -> dict:
    today = _today(settings).isoformat()
    configurable = {
        "thread_id": member.thread_id,
        "today": today,
        "proactive_instruction": instruction,
    }
    return {"configurable": configurable, "recursion_limit": settings.run_recursion_limit}


def _final_text(result: dict) -> str:
    messages = result.get("messages") or []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content and not getattr(message, "tool_calls", None):
            return content
    return "[no insight]"


def _today(settings: Settings):
    return datetime.now(ZoneInfo(settings.service_timezone)).date()


def _previous_month(today) -> tuple[int, int]:
    return (12, today.year - 1) if today.month == 1 else (today.month - 1, today.year)


def _flush_trace() -> None:
    from .observability import langfuse_client

    try:
        langfuse_client().flush()
    except Exception as exc:
        logger.warning("langfuse flush failed: %s", exc)
