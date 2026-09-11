"""One chat turn: SSE flow, the per-run and daily caps, rollback, history."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from expenso_assistant.agent import session
from expenso_assistant.agent.graph import build_graph
from expenso_assistant.config import get_settings

from .conftest import API, method_url
from .fakes import ScriptedChatModel

pytestmark = pytest.mark.usefixtures("spy_langfuse_default")


@pytest.fixture
def spy_langfuse_default(spy_langfuse):
    """Most tests just need a spy that reports 'under the cap'."""
    return spy_langfuse()


def tool_call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_1"}])


def answer(
    text: str, *, inp: int = 0, out: int = 0, cached: int = 0, reasoning: int = 0
) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": inp + out,
            "input_token_details": {"cache_read": cached},
            "output_token_details": {"reasoning": reasoning},
        },
    )


async def collect(stream) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    async for raw in stream:
        head, body, _ = raw.split("\n", 2)
        events.append((head.removeprefix("event: "), json.loads(body.removeprefix("data: "))))
    return events


async def run_turn(graph, member, text: str, *, image: str | None = None) -> list[tuple[str, dict]]:
    return await collect(
        session.stream_turn(
            graph=graph, member=member, text=text, settings=get_settings(), image=image
        )
    )


@respx.mock
async def test_happy_path_streams_steps_then_tokens_then_done(member):
    respx.get(method_url(f"{API}.get_expenses")).mock(
        return_value=httpx.Response(200, json={"message": [{"amount": 40, "category": "Food"}]})
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("get_expenses", month=3, year=2026),
            answer("You spent 40.", inp=90, out=12),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "what did I spend in March?")

    kinds = [kind for kind, _ in events]
    assert kinds[0] == "step" and kinds[-1] == "done"
    assert "token" in kinds
    assert "March 2026" in events[0][1]["text"]
    assert "".join(d["text"] for k, d in events if k == "token").strip() == "You spent 40."
    assert events[-1][1]["message_id"]


async def test_system_prompt_carries_todays_date(member):
    model = ScriptedChatModel(responses=[answer("hi")])
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "hello")

    system_text = model.last_prompt[0].content
    assert datetime.now(UTC).date().isoformat() in system_text


async def test_daily_cap_refuses_before_the_model_is_called(member, spy_langfuse):
    spy_langfuse(today_trace_count=get_settings().daily_chat_cap)
    model = ScriptedChatModel(responses=[])  # any call would IndexError
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "hi")

    assert len(events) == 1
    assert events[0][0] == "error" and events[0][1]["code"] == "daily_cap"
    assert model.calls == 0


async def test_daily_cap_check_fails_open_when_langfuse_is_down(member, spy_langfuse, caplog):
    spy_langfuse(fetch_raises=httpx.ConnectError("down"))
    model = ScriptedChatModel(responses=[answer("still works")])
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "hi")

    assert events[-1][0] == "done"
    assert "failed open" in caplog.text


async def test_wall_clock_cap_ends_in_error_with_no_partial_answer(member, monkeypatch):
    monkeypatch.setenv("RUN_WALL_CLOCK_SECONDS", "0")
    get_settings.cache_clear()
    model = ScriptedChatModel(responses=[answer("never reaches the client")])
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "hi")

    assert len(events) == 1
    assert events[0][0] == "error" and events[0][1]["code"] == "wall_clock"
    assert model.calls == 0
    assert await session.history(graph, member, get_settings()) == []


@respx.mock
async def test_tool_call_cap_ends_in_error(member, monkeypatch):
    monkeypatch.setenv("RUN_MAX_TOOL_CALLS", "1")
    get_settings.cache_clear()
    respx.get(method_url(f"{API}.get_budgets")).mock(
        return_value=httpx.Response(200, json={"message": []})
    )
    model = ScriptedChatModel(responses=[tool_call("get_budgets", month=1, year=2026)] * 4)
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "loop please")

    assert events[-1][0] == "error" and events[-1][1]["code"] == "tool_cap"
    assert "token" not in [kind for kind, _ in events]
    assert await session.history(graph, member, get_settings()) == []


@respx.mock
async def test_recursion_cap_is_the_backstop(member, monkeypatch):
    monkeypatch.setenv("RUN_RECURSION_LIMIT", "4")
    monkeypatch.setenv("RUN_MAX_TOOL_CALLS", "50")
    get_settings.cache_clear()
    respx.get(method_url(f"{API}.list_categories")).mock(
        return_value=httpx.Response(200, json={"message": ["Groceries"]})
    )
    model = ScriptedChatModel(responses=[tool_call("list_categories")] * 6)
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "loop please")

    assert events[-1][0] == "error" and events[-1][1]["code"] == "recursion"


async def test_a_failed_turn_leaves_earlier_turns_intact(member, monkeypatch):
    checkpointer = InMemorySaver()

    ok_model = ScriptedChatModel(responses=[answer("first answer")])
    await run_turn(build_graph(ok_model, checkpointer=checkpointer), member, "first question")

    monkeypatch.setenv("RUN_WALL_CLOCK_SECONDS", "0")
    get_settings.cache_clear()
    bad_model = ScriptedChatModel(responses=[answer("second answer")])
    bad_graph = build_graph(bad_model, checkpointer=checkpointer)
    await run_turn(bad_graph, member, "second question")

    history = await session.history(bad_graph, member, get_settings())
    assert [m["content"] for m in history] == ["first question", "first answer"]


@respx.mock
async def test_history_omits_tool_and_tool_call_messages(member):
    respx.get(method_url(f"{API}.get_expenses")).mock(
        return_value=httpx.Response(200, json={"message": [{"amount": 1}]})
    )
    model = ScriptedChatModel(
        responses=[tool_call("get_expenses", month=5, year=2026), answer("Spent 1.")]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "may spend?")

    history = await session.history(graph, member, get_settings())

    assert [(m["role"], m["content"]) for m in history] == [
        ("user", "may spend?"),
        ("assistant", "Spent 1."),
    ]


async def test_clear_thread_empties_history(member):
    model = ScriptedChatModel(responses=[answer("hello there")])
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "hi")

    await session.clear_thread(graph.checkpointer, member)

    assert await session.history(graph, member, get_settings()) == []


async def test_resume_with_no_pending_interrupt_is_clean(member):
    model = ScriptedChatModel(responses=[answer("hi")])
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await collect(
        session.resume_turn(graph=graph, member=member, decision={}, settings=get_settings())
    )

    assert events == [("error", {"code": "nothing_to_resume", "message": events[0][1]["message"]})]
