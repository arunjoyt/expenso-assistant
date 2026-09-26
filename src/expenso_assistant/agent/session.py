"""One chat turn: drive the graph, translate its events to SSE, and keep the
thread clean if the turn fails.

SSE event vocabulary (P6-S5 grill, Q4):

- `step`  `{text}`        — a humanized tool call, one line per action
- `token` `{text}`        — a delta of the streamed answer
- `done`  `{message_id}`  — the turn completed; commit the streamed answer
- `error` `{code, message}` — cap hit / failure; the client discards any partial

`needs_confirmation` is added in P6-S7. On any `error` the turn is rolled back to
the pre-run message set, so a failed turn leaves nothing behind in history.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from .. import tools as tool_defs
from ..auth import AuthedMember
from ..config import Settings
from ..frappe_client import FrappeClient, bind_frappe_client, reset_frappe_client
from .observability import (
    FEATURE_CHAT,
    FEATURE_RECEIPT,
    TurnTrace,
    score_receipt_accuracy,
    within_daily_token_cap,
)
from .state import RunContext

logger = logging.getLogger(__name__)

_HUMANIZE = {
    "get_expenses": "Reading expenses",
    "get_income": "Reading income",
    "get_analytics": "Checking the monthly summary",
    "get_budgets": "Reading budgets",
    "list_categories": "Listing categories",
    "list_sources": "Listing income sources",
}

_MONTHS = (
    "January February March April May June July August September October November December".split()
)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def humanize(tool_name: str, args: dict | None) -> str:
    base = _HUMANIZE.get(tool_name, tool_name.replace("_", " ").capitalize())
    month, year = (args or {}).get("month"), (args or {}).get("year")
    if isinstance(month, int) and 1 <= month <= 12:
        span = _MONTHS[month - 1] + (f" {year}" if year else "")
        return f"{base} for {span}"
    return f"{base}…"


async def stream_turn(
    *, graph, member: AuthedMember, text: str, settings: Settings, image: str | None = None
) -> AsyncIterator[str]:
    # No fail-open here (unlike the old Langfuse-backed cap): the counter's
    # storage is the checkpointer's own Postgres, already required for the
    # turn to run at all, so a check failure fails the turn the same way a
    # checkpointer failure downstream would (ADR 0008's 2026-09-11 update).
    try:
        allowed = within_daily_token_cap(member.email)
    except Exception:
        logger.exception("daily cap check failed")
        yield sse("error", {"code": "internal", "message": _ERROR_MESSAGES["internal"]})
        return
    if not allowed:
        yield sse("error", {"code": "daily_cap", "message": "You've reached today's usage limit."})
        return

    config = _run_config(member, settings)
    context = _run_context(settings, image=image)
    entry_method = "receipt" if image else "assistant"
    feature = FEATURE_RECEIPT if image else FEATURE_CHAT

    # A new message while a confirm card is open means the member moved on — the
    # thread is linear. Drop the unconfirmed proposal and carry on (P6-S7).
    if await discard_pending(graph, config):
        yield sse("step", {"text": "Discarded the unconfirmed changes"})

    if image:
        yield sse("step", {"text": "Reading the receipt…"})

    pre_ids = await _message_ids(graph, config)
    trace = TurnTrace.start(
        user_id=member.email, session_id=member.thread_id, user_input=text, feature=feature
    )
    config["callbacks"] = [trace.callback]

    # The graph's tool node reaches Frappe as this Member (bearer passthrough),
    # same contextvar the FastMCP adapter binds.
    client_token = bind_frappe_client(FrappeClient(member.token))
    entry_token = tool_defs.bind_entry_method(entry_method)
    try:
        human_text = _human_text(text, image)
        inputs = {
            "messages": [HumanMessage(human_text)],
            "entry_method": entry_method,
        }
        async for event in _drive(graph, inputs, config, context, trace, pre_ids, settings):
            yield event
    finally:
        tool_defs.reset_entry_method(entry_token)
        reset_frappe_client(client_token)
        _flush_trace()  # last, so the trace's final output/cost updates go too


async def resume_turn(
    *, graph, member: AuthedMember, decision: dict, settings: Settings
) -> AsyncIterator[str]:
    """The member's decision on a pending confirm card. Executes the approved
    subset (P6-S7) and streams the continuation on a fresh SSE leg — its own
    Langfuse trace, no chat-cap re-check (the turn already passed it)."""
    config = _run_config(member, settings)
    card = await pending_card(graph, config)
    if card is None:
        yield sse("error", {"code": "nothing_to_resume", "message": "No pending confirmation."})
        return

    trace = TurnTrace.start(
        user_id=member.email, session_id=member.thread_id, user_input=json.dumps(decision)
    )
    config["callbacks"] = [trace.callback]
    # The approved writes run on this leg, so they must carry the turn's own
    # entry method ("receipt" survives the pause in state, P7-S1).
    state = await graph.aget_state(config)
    entry_method = (state.values or {}).get("entry_method", "assistant")
    if entry_method == "receipt":
        score_receipt_accuracy(card, decision, trace)
    context = _run_context(settings)
    client_token = bind_frappe_client(FrappeClient(member.token))
    entry_token = tool_defs.bind_entry_method(entry_method)
    try:
        resume = Command(resume=decision)
        async for event in _drive(graph, resume, config, context, trace, None, settings):
            yield event
    finally:
        tool_defs.reset_entry_method(entry_token)
        reset_frappe_client(client_token)
        _flush_trace()


async def history(graph, member: AuthedMember, settings: Settings) -> list[dict]:
    state = await graph.aget_state(_run_config(member, settings))
    messages = (state.values or {}).get("messages", [])
    out: list[dict] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            out.append({"id": message.id, "role": "user", "content": message.content})
        elif isinstance(message, AIMessage) and message.content and not message.tool_calls:
            entry = {"id": message.id, "role": "assistant", "content": message.content}
            # A proactive Insight (P7-S2) is tagged in `additional_kwargs` by
            # `middleware.py`'s `ModelCallShaping`; sparse and optional, same pattern as
            # P7-S1's `edits` — an ordinary reply carries neither key.
            if message.additional_kwargs.get("kind") == "insight":
                entry["kind"] = "insight"
                entry["posted_at"] = message.additional_kwargs.get("posted_at")
            out.append(entry)
    return out


async def clear_thread(checkpointer, member: AuthedMember) -> None:
    await checkpointer.adelete_thread(member.thread_id)


# --- internals ------------------------------------------------------------


def _run_config(member: AuthedMember, settings: Settings) -> dict:
    return {
        "configurable": {"thread_id": member.thread_id},
        "recursion_limit": settings.run_recursion_limit,
    }


def _run_context(settings: Settings, *, image: str | None = None) -> RunContext:
    # The receipt image rides in the run context (P7-S1), not in graph state,
    # precisely so it is never checkpointed.
    today = datetime.now(ZoneInfo(settings.service_timezone)).date()
    return RunContext(today=today, receipt_image=image)


def _human_text(text: str, image: str | None) -> str:
    """What gets checkpointed for this turn's HumanMessage — the image itself
    never does (P7-S1). A short marker, plus any caption the Member typed."""
    if not image:
        return text
    marker = "[Attached a photo]"
    return f"{marker} {text}" if text else marker


async def _drive(
    graph, inputs, config, context: RunContext, trace: TurnTrace, pre_ids, settings: Settings
) -> AsyncIterator[str]:
    """Streams the run with LangGraph's own stream modes (ADR 0010): `tasks`
    for each tool as it runs, `messages` for answer tokens, `updates` for the
    confirm card and the final reply's id."""
    run = _RunStream()
    deadline = asyncio.get_event_loop().time() + settings.run_wall_clock_seconds
    parts = graph.astream(
        inputs,
        config,
        context=context,
        stream_mode=_STREAM_MODES,
        durability=_DURABILITY,
        version="v2",
    )
    try:
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError
            try:
                part = await asyncio.wait_for(parts.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            for event in run.read(part):
                yield event
    except (TimeoutError, GraphRecursionError, ToolCallLimitExceededError) as exc:
        code = _ERROR_CODES[type(exc)]
        await _rollback(graph, config, pre_ids)
        trace.fail(code)
        yield sse("error", {"code": code, "message": _ERROR_MESSAGES[code]})
        return
    except Exception:
        logger.exception("chat turn failed")
        await _rollback(graph, config, pre_ids)
        trace.fail("internal")
        yield sse("error", {"code": "internal", "message": _ERROR_MESSAGES["internal"]})
        return

    if run.card is not None:
        trace.finish(output="[needs confirmation]")
        yield sse("needs_confirmation", run.card)
        return

    trace.finish(output="".join(run.answer))
    yield sse("done", {"message_id": run.message_id})


_STREAM_MODES = ["tasks", "messages", "updates"]
# LangGraph's default, stated on purpose (ADR 0010): each step is saved while
# the next runs. "exit" would save only when the run ends — fewer Postgres
# writes, but a crash mid-resume would lose the record of a committed write,
# reopen the card, and let a re-confirm duplicate the row.
_DURABILITY = "async"


class _RunStream:
    """Turns LangGraph v2 stream parts into SSE events, keeping what the end
    of the leg needs: the answer text, the confirm card, the reply's id."""

    def __init__(self):
        self.answer: list[str] = []
        self.card: dict | None = None
        self.message_id: str | None = None

    def read(self, part: dict) -> list[str]:
        kind, data = part["type"], part["data"]
        if kind == "tasks" and data.get("name") == "tools" and "input" in data:
            return [
                sse("step", {"text": humanize(c["name"], c.get("args"))}) for c in data["input"]
            ]
        if kind == "messages":
            message, meta = data
            piece = _chunk_text(message) if meta.get("langgraph_node") == "model" else ""
            if piece:
                self.answer.append(piece)
                return [sse("token", {"text": piece})]
        if kind == "updates":
            self._read_update(data)
        return []

    def _read_update(self, data: dict) -> None:
        if interrupts := data.get("__interrupt__"):
            self.card = interrupts[0].value
        reply = ((data.get("model") or {}).get("messages") or [None])[-1]
        if isinstance(reply, AIMessage) and reply.content and not reply.tool_calls:
            self.message_id = reply.id


_ERROR_CODES = {
    TimeoutError: "wall_clock",
    GraphRecursionError: "recursion",
    ToolCallLimitExceededError: "tool_cap",
}

_ERROR_MESSAGES = {
    "wall_clock": "The assistant took too long and stopped.",
    "recursion": "The assistant got stuck and stopped.",
    "tool_cap": "The assistant tried too many steps and stopped.",
    "internal": "The assistant hit an error.",
}


def _flush_trace() -> None:
    from .observability import langfuse_client

    try:
        langfuse_client().flush()
    except Exception as exc:
        logger.warning("langfuse flush failed: %s", exc)


def _chunk_text(chunk) -> str:
    content = getattr(chunk, "content", "")
    return content if isinstance(content, str) else ""


async def _message_ids(graph, config) -> set[str]:
    state = await graph.aget_state(config)
    return {m.id for m in (state.values or {}).get("messages", [])}


async def pending_card(graph, config) -> dict | None:
    """The confirm-card payload if the graph is paused on an `interrupt`."""
    state = await graph.aget_state(config)
    for task in state.tasks:
        for interrupt in task.interrupts:
            return interrupt.value
    return None


async def discard_pending(graph, config) -> bool:
    """Throw away an unconfirmed proposal: drop the trailing AI message whose
    tool calls were never answered. Returns whether there was one.

    Found from state, not from the paused interrupt: a deploy that changes the
    graph's node names loses the paused task but keeps the message, and
    OpenAI rejects a tool call with no `ToolMessage` — every later turn in the
    thread would fail (ADR 0010, spike results)."""
    state = await graph.aget_state(config)
    messages = (state.values or {}).get("messages", [])
    if not messages or not getattr(messages[-1], "tool_calls", None):
        return False
    # Any task this leaves pending is dropped when the next run's input arrives.
    await graph.aupdate_state(
        config, {"messages": [RemoveMessage(id=messages[-1].id)]}, as_node="model"
    )
    return True


async def _rollback(graph, config, pre_ids: set[str] | None) -> None:
    """Drop every message this turn added, so history reads as if it never ran.
    `pre_ids=None` (the resume leg) skips rollback — its writes are real rows."""
    if pre_ids is None:
        return
    try:
        state = await graph.aget_state(config)
        stale = [
            RemoveMessage(id=m.id)
            for m in (state.values or {}).get("messages", [])
            if m.id not in pre_ids
        ]
        if stale:
            await graph.aupdate_state(config, {"messages": stale})
    except Exception as exc:
        logger.warning("turn rollback failed: %s", exc)
