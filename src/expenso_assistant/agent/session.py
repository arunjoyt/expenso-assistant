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

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from .. import tools as tool_defs
from ..auth import AuthedMember
from ..config import Settings
from ..frappe_client import FrappeClient, bind_frappe_client, reset_frappe_client
from .graph import ToolCapExceeded
from .observability import FEATURE_CHAT, FEATURE_RECEIPT, TurnTrace, within_daily_chat_cap

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
    if not within_daily_chat_cap(member.email):
        yield sse("error", {"code": "daily_cap", "message": "You've reached today's chat limit."})
        return

    config = _run_config(member, settings, image=image)
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
            "tool_call_count": 0,
            "entry_method": entry_method,
        }
        async for event in _drive(graph, inputs, config, trace, pre_ids, settings):
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
    if await pending_card(graph, config) is None:
        yield sse("error", {"code": "nothing_to_resume", "message": "No pending confirmation."})
        return

    trace = TurnTrace.start(
        user_id=member.email, session_id=member.thread_id, user_input=json.dumps(decision)
    )
    config["callbacks"] = [trace.callback]
    # propose.py's _apply reads this to post receipt_accuracy_* scores here —
    # the comparison only exists once the Member confirms/edits (P7-S1).
    config["configurable"]["trace"] = trace
    client_token = bind_frappe_client(FrappeClient(member.token))
    entry_token = tool_defs.bind_entry_method("assistant")
    try:
        async for event in _drive(graph, Command(resume=decision), config, trace, None, settings):
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
            # `graph.py`'s `agent_node`; sparse and optional, same pattern as
            # P7-S1's `edits` — an ordinary reply carries neither key.
            if message.additional_kwargs.get("kind") == "insight":
                entry["kind"] = "insight"
                entry["posted_at"] = message.additional_kwargs.get("posted_at")
            out.append(entry)
    return out


async def clear_thread(checkpointer, member: AuthedMember) -> None:
    await checkpointer.adelete_thread(member.thread_id)


# --- internals ------------------------------------------------------------


def _run_config(member: AuthedMember, settings: Settings, *, image: str | None = None) -> dict:
    # `today` as an ISO string, not a date — it rides in `configurable`, which
    # the Postgres checkpointer serializes as JSON. `receipt_image` rides here
    # too (P7-S1) rather than in graph state, precisely so it is never
    # checkpointed — `config` is per-run, `state` is what Postgres persists.
    today = datetime.now(ZoneInfo(settings.service_timezone)).date().isoformat()
    configurable = {"thread_id": member.thread_id, "today": today}
    if image:
        configurable["receipt_image"] = image
    return {"configurable": configurable, "recursion_limit": settings.run_recursion_limit}


def _human_text(text: str, image: str | None) -> str:
    """What gets checkpointed for this turn's HumanMessage — the image itself
    never does (P7-S1). A short marker, plus any caption the Member typed."""
    if not image:
        return text
    marker = "[Attached a photo]"
    return f"{marker} {text}" if text else marker


async def _drive(
    graph, inputs, config, trace: TurnTrace, pre_ids, settings: Settings
) -> AsyncIterator[str]:
    answer: list[str] = []
    deadline = asyncio.get_event_loop().time() + settings.run_wall_clock_seconds
    events = graph.astream_events(inputs, config, version="v2")
    try:
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError
            try:
                event = await asyncio.wait_for(events.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            kind = event["event"]
            if kind == "on_tool_start":
                yield sse("step", {"text": humanize(event["name"], event["data"].get("input"))})
            elif kind == "on_chat_model_stream":
                piece = _chunk_text(event["data"].get("chunk"))
                if piece:
                    answer.append(piece)
                    yield sse("token", {"text": piece})
    except (TimeoutError, GraphRecursionError, ToolCapExceeded) as exc:
        code = _ERROR_CODES[type(exc)]
        await _rollback(graph, config, pre_ids)
        trace.fail(code)
        yield sse("error", {"code": code, "message": _ERROR_MESSAGES[code]})
        return
    except Exception:
        logger.exception("chat turn failed")
        await _rollback(graph, config, pre_ids)
        trace.fail("internal")
        yield sse("error", {"code": "internal", "message": "The assistant hit an error."})
        return

    card = await pending_card(graph, config)
    if card is not None:
        trace.finish(output="[needs confirmation]")
        yield sse("needs_confirmation", card)
        return

    final = "".join(answer)
    trace.finish(output=final)
    yield sse("done", {"message_id": await _latest_ai_id(graph, config)})


_ERROR_CODES = {
    TimeoutError: "wall_clock",
    GraphRecursionError: "recursion",
    ToolCapExceeded: "tool_cap",
}

_ERROR_MESSAGES = {
    "wall_clock": "The assistant took too long and stopped.",
    "recursion": "The assistant got stuck and stopped.",
    "tool_cap": "The assistant tried too many steps and stopped.",
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


async def _latest_ai_id(graph, config) -> str | None:
    state = await graph.aget_state(config)
    for message in reversed((state.values or {}).get("messages", [])):
        if isinstance(message, AIMessage) and message.content:
            return message.id
    return None


async def pending_card(graph, config) -> dict | None:
    """The confirm-card payload if the graph is paused on an `interrupt`."""
    state = await graph.aget_state(config)
    for task in state.tasks:
        for interrupt in task.interrupts:
            return interrupt.value
    return None


async def discard_pending(graph, config) -> bool:
    """Throw away an unconfirmed proposal: drop the AI message whose write
    tool-calls opened the card, and re-route from `agent` so `state.next` clears.
    Returns whether there was one."""
    if await pending_card(graph, config) is None:
        return False
    state = await graph.aget_state(config)
    messages = (state.values or {}).get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            await graph.aupdate_state(
                config, {"messages": [RemoveMessage(id=message.id)]}, as_node="agent"
            )
            return True
    return False


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
