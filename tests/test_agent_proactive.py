"""P7-S2 — proactive runs: no dedup, no proposals (structural, via READ_TOOLS
binding), a deterministic pre-check gates `budget_drift`'s LLM call, and a
stale pending proposal is discarded before an Insight is posted."""

from __future__ import annotations

import httpx
import pytest
import respx
from langgraph.checkpoint.memory import InMemorySaver

from expenso_assistant import tools as tool_defs
from expenso_assistant.agent import proactive, session
from expenso_assistant.agent.graph import build_graph
from expenso_assistant.config import get_settings

from .conftest import API, method_url
from .fakes import ScriptedChatModel
from .test_agent_session import answer, run_turn
from .test_agent_writes import multi_call

pytestmark = pytest.mark.usefixtures("spy_langfuse_default")


@pytest.fixture
def spy_langfuse_default(spy_langfuse):
    return spy_langfuse()


def _config(member) -> dict:
    return {"configurable": {"thread_id": member.thread_id}}


def _analytics(categories: list[dict]) -> httpx.Response:
    return httpx.Response(
        200, json={"message": {"total": 0, "income_total": 0, "categories": categories}}
    )


async def test_monthly_summary_always_calls_the_model_and_posts_a_tagged_insight(member):
    model = ScriptedChatModel(responses=[answer("August was quiet — you spent 100.")])
    graph = build_graph(model, checkpointer=InMemorySaver(), tools=tool_defs.READ_TOOLS)
    settings = get_settings()

    await proactive.run_proactive_job(
        job=proactive.JOB_MONTHLY_SUMMARY, member=member, graph=graph, settings=settings
    )

    assert model.calls == 1
    history = await session.history(graph, member, settings)
    assert len(history) == 1
    assert history[0]["role"] == "assistant"
    assert history[0]["kind"] == "insight"
    assert history[0]["posted_at"]
    assert history[0]["content"] == "August was quiet — you spent 100."


@respx.mock
async def test_budget_drift_skips_the_model_when_nothing_is_over_budget(member):
    respx.get(method_url(f"{API}.get_analytics")).mock(
        return_value=_analytics([{"name": "Dining", "budget_status": "Normal"}])
    )
    model = ScriptedChatModel(responses=[])  # any call would IndexError
    graph = build_graph(model, checkpointer=InMemorySaver(), tools=tool_defs.READ_TOOLS)
    settings = get_settings()

    await proactive.run_proactive_job(
        job=proactive.JOB_BUDGET_DRIFT, member=member, graph=graph, settings=settings
    )

    assert model.calls == 0
    assert await session.history(graph, member, settings) == []


@respx.mock
async def test_budget_drift_calls_the_model_naming_only_the_crossed_categories(member):
    respx.get(method_url(f"{API}.get_analytics")).mock(
        return_value=_analytics(
            [
                {"name": "Dining", "budget_status": "Exceeded"},
                {"name": "Groceries", "budget_status": "Normal"},
                {"name": "Travel", "budget_status": "Warning"},
            ]
        )
    )
    model = ScriptedChatModel(responses=[answer("Dining and Travel are over budget.")])
    graph = build_graph(model, checkpointer=InMemorySaver(), tools=tool_defs.READ_TOOLS)
    settings = get_settings()

    await proactive.run_proactive_job(
        job=proactive.JOB_BUDGET_DRIFT, member=member, graph=graph, settings=settings
    )

    assert model.calls == 1
    instruction = model.last_prompt[-1].content
    assert "Dining" in instruction and "Travel" in instruction
    assert "Groceries" not in instruction
    history = await session.history(graph, member, settings)
    assert history[0]["kind"] == "insight"


@respx.mock
async def test_budget_drift_reruns_every_week_with_no_dedup(member):
    respx.get(method_url(f"{API}.get_analytics")).mock(
        return_value=_analytics([{"name": "Dining", "budget_status": "Exceeded"}])
    )
    model = ScriptedChatModel(
        responses=[answer("Still over on Dining."), answer("Still over on Dining.")]
    )
    graph = build_graph(model, checkpointer=InMemorySaver(), tools=tool_defs.READ_TOOLS)
    settings = get_settings()

    await proactive.run_proactive_job(
        job=proactive.JOB_BUDGET_DRIFT, member=member, graph=graph, settings=settings
    )
    await proactive.run_proactive_job(
        job=proactive.JOB_BUDGET_DRIFT, member=member, graph=graph, settings=settings
    )

    history = await session.history(graph, member, settings)
    assert len(history) == 2
    assert all(entry["kind"] == "insight" for entry in history)


async def test_unknown_job_is_a_no_op(member):
    model = ScriptedChatModel(responses=[])
    graph = build_graph(model, checkpointer=InMemorySaver(), tools=tool_defs.READ_TOOLS)
    settings = get_settings()

    await proactive.run_proactive_job(
        job="not-a-real-job", member=member, graph=graph, settings=settings
    )

    assert model.calls == 0
    assert await session.history(graph, member, settings) == []


@respx.mock
async def test_pending_proposal_is_discarded_before_the_insight_is_posted(member):
    checkpointer = InMemorySaver()
    write_model = ScriptedChatModel(responses=[multi_call(("add_category", {"name": "Travel"}))])
    interactive_graph = build_graph(write_model, checkpointer=checkpointer)
    await run_turn(interactive_graph, member, "add a travel category")
    assert await session.pending_card(interactive_graph, _config(member)) is not None

    insight_model = ScriptedChatModel(responses=[answer("Here's your summary.")])
    read_only_graph = build_graph(
        insight_model, checkpointer=checkpointer, tools=tool_defs.READ_TOOLS
    )
    settings = get_settings()

    await proactive.run_proactive_job(
        job=proactive.JOB_MONTHLY_SUMMARY, member=member, graph=read_only_graph, settings=settings
    )

    assert await session.pending_card(read_only_graph, _config(member)) is None
    history = await session.history(read_only_graph, member, settings)
    assert any(entry.get("kind") == "insight" for entry in history)
