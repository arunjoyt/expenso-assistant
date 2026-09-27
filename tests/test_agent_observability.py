"""LangSmith tracing (ADR 0011): one tagged trace per turn, receipt images
masked before upload, and the daily token count."""

from __future__ import annotations

import json

import pytest
from langchain_core.tracers import LangChainTracer
from langgraph.checkpoint.memory import InMemorySaver

from expenso_assistant.agent import observability
from expenso_assistant.agent.graph import build_graph
from expenso_assistant.agent.observability import IMAGE_PLACEHOLDER, mask_images
from expenso_assistant.config import get_settings

from .fakes import ScriptedChatModel
from .test_agent_session import answer, run_turn, tool_call

IMAGE = "data:image/jpeg;base64,Zm9vYmFy"


@pytest.fixture
def spy(spy_tracing):
    return spy_tracing()


async def test_one_trace_per_turn_tagged_for_the_member(member, spy):
    model = ScriptedChatModel(responses=[answer("Hi.")])
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "hello")

    [trace] = spy.recorded
    assert trace.name == "chat-turn"
    assert "feature:chat" in trace.tags
    assert trace.extra["metadata"]["feature"] == "chat"
    assert trace.extra["metadata"]["user_id"] == member.email
    assert trace.extra["metadata"]["session_id"] == member.thread_id
    assert trace.error is None
    assert trace.outputs["messages"][-1].content.strip() == "Hi."


async def test_the_model_call_is_nested_under_the_turn(member, spy):
    model = ScriptedChatModel(responses=[answer("Hi.", inp=1_000, out=50)])
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "hello")

    [llm_run] = [r for r in _all_runs(spy.recorded[0]) if r.run_type == "llm"]
    usage = llm_run.outputs["generations"][0][0]["message"]["kwargs"]["usage_metadata"]
    assert usage["input_tokens"] == 1_000  # what LangSmith prices the call from


async def test_a_failed_turn_is_recorded_as_an_error_on_its_trace(member, spy):
    model = ScriptedChatModel(responses=[])  # the model call raises
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "hello")

    assert events[-1][1]["code"] == "internal"
    [trace] = spy.recorded
    assert trace.error


async def test_daily_cap_checked_before_this_runs_own_trace_opens(member, spy, spy_token_store):
    spy_token_store(today_tokens=get_settings().daily_token_cap)
    model = ScriptedChatModel(responses=[])
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "hello")

    assert spy.recorded == []


async def test_generation_tokens_are_recorded_against_the_daily_total(member, spy_token_store):
    spy = spy_token_store()
    model = ScriptedChatModel(responses=[answer("Hi.", inp=1_000, out=50)])
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "hello")

    assert spy.added == [(member.email, 1_050)]
    assert spy.get(member.email, None) == 1_050


async def test_tool_call_only_generation_with_no_usage_logs_a_warning(member, spy, caplog):
    # expenso-assistant#4: a live trace comparison found empty usage on
    # generations ending in a tool call with no final text.
    model = ScriptedChatModel(responses=[tool_call("create_expense", amount=5)])
    graph = build_graph(model, checkpointer=InMemorySaver())

    with caplog.at_level("WARNING", logger="expenso_assistant.agent.observability"):
        events = await run_turn(graph, member, "add a coffee expense of 5")

    assert events[-1][0] == "needs_confirmation"
    [record] = [r for r in caplog.records if "zero usage on LLM response" in r.message]
    assert "tool_calls=True" in record.message


# --- receipt images never leave the service --------------------------------


async def test_receipt_image_is_masked_in_every_run_of_the_trace(member, spy):
    model = ScriptedChatModel(responses=[answer("Not a receipt.")])
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "", image=IMAGE)

    runs = list(_all_runs(spy.recorded[0]))
    raw = json.dumps([r.inputs for r in runs], default=str)
    masked = json.dumps([mask_images(r.inputs) for r in runs], default=str)
    assert IMAGE in raw  # the model really was sent the image
    assert "base64," not in masked
    assert IMAGE_PLACEHOLDER in masked


def test_mask_images_keeps_everything_but_image_data():
    inputs = {
        "messages": [
            [
                {
                    "content": [
                        {"type": "text", "text": "12 at the grocer"},
                        {"type": "image_url", "image_url": {"url": IMAGE}},
                    ]
                }
            ]
        ],
        "amount": 12,
    }

    masked = mask_images(inputs)

    blocks = masked["messages"][0][0]["content"]
    assert blocks[0] == {"type": "text", "text": "12 at the grocer"}
    assert blocks[1]["image_url"]["url"] == IMAGE_PLACEHOLDER
    assert masked["amount"] == 12


# --- configuration ---------------------------------------------------------


def test_tracing_is_off_without_an_api_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    get_settings.cache_clear()
    observability.reset_langsmith_client()

    assert observability.tracers() == []
    observability.flush_traces()  # a no-op, not an error


def test_tracer_sends_to_the_eu_region_with_images_masked(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test")
    get_settings.cache_clear()
    observability.reset_langsmith_client()
    try:
        [tracer] = observability.tracers()
        assert isinstance(tracer, LangChainTracer)
        assert tracer.project_name == "expenso-assistant"
        assert tracer.client.api_url == "https://eu.api.smith.langchain.com"
        assert tracer.client._hide_inputs is mask_images
    finally:
        observability.reset_langsmith_client()


def _all_runs(run):
    yield run
    for child in run.child_runs:
        yield from _all_runs(child)
