"""Langfuse tagging, the explicit-cost generation, and the daily-cap query."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from expenso_assistant.agent.graph import build_graph
from expenso_assistant.config import get_settings

from .fakes import ScriptedChatModel
from .test_agent_session import answer, run_turn


@pytest.fixture
def spy(spy_langfuse):
    return spy_langfuse()


async def test_one_trace_per_turn_tagged_for_the_member(member, spy):
    model = ScriptedChatModel(responses=[answer("Hi.", inp=1_000_000, out=1_000_000)])
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "hello")

    assert len(spy.traces) == 1
    trace = spy.traces[0]
    assert trace.init["user_id"] == member.email
    assert trace.init["session_id"] == member.thread_id
    assert trace.init["metadata"]["feature"] == "chat"
    assert "feature:chat" in trace.init["tags"]
    assert trace.updates[-1]["output"].strip() == "Hi."


async def test_generation_cost_is_the_services_number(member, spy):
    model = ScriptedChatModel(responses=[answer("Hi.", inp=1_000_000, out=1_000_000)])
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "hello")

    generations = spy.traces[0].generations
    assert len(generations) == 1
    # 1M uncached input + 1M output on gpt-4o-mini = 0.15 + 0.60
    assert round(generations[0]["cost_details"]["total"], 6) == 0.75
    assert generations[0]["usage_details"]["input"] == 1_000_000


async def test_generation_cost_discounts_cache_and_bills_reasoning_as_output(member, spy):
    model = ScriptedChatModel(
        responses=[answer("Hi.", inp=1_000_000, cached=1_000_000, out=0, reasoning=1_000_000)]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "hello")

    # all input cached (0.075) + reasoning as output (0.60)
    assert round(spy.traces[0].generations[0]["cost_details"]["total"], 6) == 0.675


async def test_daily_cap_checked_before_this_runs_own_trace_opens(member, spy, spy_token_store):
    spy_token_store(today_tokens=get_settings().daily_token_cap)
    model = ScriptedChatModel(responses=[])
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "hello")

    assert spy.traces == []


async def test_generation_tokens_are_recorded_against_the_daily_total(member, spy_token_store):
    spy = spy_token_store()
    model = ScriptedChatModel(responses=[answer("Hi.", inp=1_000, out=50)])
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "hello")

    assert spy.added == [(member.email, 1_050)]
    assert spy.get(member.email, None) == 1_050
