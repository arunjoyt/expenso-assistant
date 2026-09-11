"""P7-S1 — receipts: image threading, entry_method capture, accuracy scoring.

The scripted model keeps OpenAI (and any real vision call) out — these tests
assert on the *shape* of what the agent does with an attached image, not on
extraction quality.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from expenso_assistant.agent import session
from expenso_assistant.agent.graph import _with_receipt_image, build_graph
from expenso_assistant.config import get_settings

from .conftest import API, method_url
from .fakes import ScriptedChatModel
from .test_agent_session import answer, collect, run_turn
from .test_agent_writes import multi_call

pytestmark = pytest.mark.usefixtures("spy_langfuse_default")

IMAGE = "data:image/jpeg;base64,Zm9vYmFy"


@pytest.fixture
def spy_langfuse_default(spy_langfuse):
    return spy_langfuse()


async def _resume(graph, member, decision) -> list[tuple[str, dict]]:
    return await collect(
        session.resume_turn(graph=graph, member=member, decision=decision, settings=get_settings())
    )


# --- input shape drives tagging ------------------------------------------


async def test_image_attached_binds_receipt_entry_method_and_feature(member, spy_langfuse):
    spy = spy_langfuse()
    model = ScriptedChatModel(
        responses=[multi_call(("create_expense", {"amount": 12, "category": "Groceries"}))]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "", image=IMAGE)

    assert spy.traces[0].init["metadata"]["feature"] == "receipt"
    assert "feature:receipt" in spy.traces[0].init["tags"]
    (action,) = events[-1][1]["actions"]
    assert action["tool"] == "create_expense"  # entry_method is internal, not in the public card


async def test_non_image_turn_still_tags_chat(member, spy_langfuse):
    spy = spy_langfuse()
    model = ScriptedChatModel(responses=[answer("hi")])
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "hello")

    assert spy.traces[0].init["metadata"]["feature"] == "chat"


async def test_reading_receipt_step_precedes_everything(member):
    model = ScriptedChatModel(responses=[answer("Not a receipt — what would you like to do?")])
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "", image=IMAGE)

    assert events[0] == ("step", {"text": "Reading the receipt…"})


async def test_receipt_turns_share_the_chat_daily_cap(member, spy_langfuse):
    spy_langfuse(today_trace_counts={"feature:chat": get_settings().daily_chat_cap})
    model = ScriptedChatModel(responses=[])  # any call would IndexError
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "", image=IMAGE)

    assert events == [("error", {"code": "daily_cap", "message": events[0][1]["message"]})]
    assert model.calls == 0


# --- the image never touches checkpointed state ---------------------------


def test_with_receipt_image_builds_a_transient_multimodal_copy():
    messages = [HumanMessage("[Attached a photo]")]
    spliced = _with_receipt_image(messages, IMAGE)

    assert spliced is not messages
    assert isinstance(spliced[0].content, list)
    assert spliced[0].content[0] == {"type": "text", "text": "[Attached a photo]"}
    assert spliced[0].content[1] == {"type": "image_url", "image_url": {"url": IMAGE}}
    # the original, checkpointable message is untouched
    assert messages[0].content == "[Attached a photo]"


def test_with_receipt_image_is_a_noop_without_an_image():
    messages = [HumanMessage("hello")]
    assert _with_receipt_image(messages, None) is messages


async def test_checkpointed_history_never_carries_the_image(member):
    model = ScriptedChatModel(
        responses=[multi_call(("create_expense", {"amount": 12, "category": "Groceries"}))]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "lunch", image=IMAGE)

    state = await graph.aget_state(session._run_config(member, get_settings()))
    for message in (state.values or {}).get("messages", []):
        assert IMAGE not in str(message.content)


async def test_thread_marker_includes_caption(member):
    model = ScriptedChatModel(responses=[answer("Got it.")])
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "lunch with the team", image=IMAGE)

    history = await session.history(graph, member, get_settings())
    assert history[0]["content"] == "[Attached a photo] lunch with the team"


async def test_thread_marker_alone_with_no_caption(member):
    model = ScriptedChatModel(responses=[answer("Got it.")])
    graph = build_graph(model, checkpointer=InMemorySaver())

    await run_turn(graph, member, "", image=IMAGE)

    history = await session.history(graph, member, get_settings())
    assert history[0]["content"] == "[Attached a photo]"


# --- the confirm card + resume + accuracy scoring -------------------------


@respx.mock
async def test_confirmed_receipt_create_stamps_entry_method_receipt(member):
    create = respx.post(method_url(f"{API}.create_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "EXP-99"}})
    )
    model = ScriptedChatModel(
        responses=[
            multi_call(("create_expense", {"amount": 12, "category": "Groceries"})),
            answer("Added it."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "", image=IMAGE)

    await _resume(graph, member, {"selected": ["w0"]})

    body = json.loads(create.calls.last.request.read())
    assert body["entry_method"] == "receipt"


@respx.mock
async def test_unedited_confirm_scores_full_agreement_on_the_resume_trace(member, spy_langfuse):
    spy = spy_langfuse()
    respx.post(method_url(f"{API}.create_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "EXP-99"}})
    )
    model = ScriptedChatModel(
        responses=[
            multi_call(
                (
                    "create_expense",
                    {"amount": 12, "date": "2026-03-10", "category": "Groceries", "notes": "milk"},
                )
            ),
            answer("Added it."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "", image=IMAGE)
    traces_before_resume = list(spy.traces)

    await _resume(graph, member, {"selected": ["w0"]})

    resume_trace = next(t for t in spy.traces if t not in traces_before_resume)
    scores = {s["name"]: s["value"] for s in resume_trace.scores}
    assert scores == {
        "receipt_accuracy_amount": 1.0,
        "receipt_accuracy_date": 1.0,
        "receipt_accuracy_category": 1.0,
        "receipt_accuracy_notes": 1.0,
    }
    # the trace that ran the vision call gets none of these
    assert traces_before_resume[0].scores == []


@respx.mock
async def test_edited_amount_scores_zero_only_on_that_field(member, spy_langfuse):
    spy = spy_langfuse()
    respx.post(method_url(f"{API}.create_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "EXP-99"}})
    )
    model = ScriptedChatModel(
        responses=[
            multi_call(("create_expense", {"amount": 12, "category": "Groceries"})),
            answer("Added it."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "", image=IMAGE)

    await _resume(graph, member, {"selected": ["w0"], "edits": {"w0": {"amount": 15}}})

    scores = {s["name"]: s["value"] for s in spy.traces[-1].scores}
    assert scores["receipt_accuracy_amount"] == 0.0
    assert scores["receipt_accuracy_category"] == 1.0


@respx.mock
async def test_edited_value_is_what_actually_gets_saved(member):
    create = respx.post(method_url(f"{API}.create_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "EXP-99"}})
    )
    model = ScriptedChatModel(
        responses=[
            multi_call(("create_expense", {"amount": 12, "category": "Groceries"})),
            answer("Added it."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "", image=IMAGE)

    await _resume(graph, member, {"selected": ["w0"], "edits": {"w0": {"amount": 15}}})

    body = json.loads(create.calls.last.request.read())
    assert body["amount"] == 15


async def test_rejected_proposal_scores_nothing(member, spy_langfuse):
    spy = spy_langfuse()
    model = ScriptedChatModel(
        responses=[
            multi_call(("create_expense", {"amount": 12, "category": "Groceries"})),
            answer("Okay, not added."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "", image=IMAGE)

    await _resume(graph, member, {"selected": []})

    assert all(t.scores == [] for t in spy.traces)


@respx.mock
async def test_null_proposed_field_is_excluded_from_scoring(member, spy_langfuse):
    spy = spy_langfuse()
    respx.post(method_url(f"{API}.create_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "EXP-99"}})
    )
    model = ScriptedChatModel(
        responses=[
            multi_call(("create_expense", {"amount": 12})),  # no category, no date, no notes
            answer("Added it."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "", image=IMAGE)

    await _resume(graph, member, {"selected": ["w0"]})

    scores = {s["name"] for s in spy.traces[-1].scores}
    assert scores == {"receipt_accuracy_amount"}


@respx.mock
async def test_non_receipt_write_is_never_scored(member, spy_langfuse):
    """A plain (non-image) 'add this expense' create is entry_method=assistant
    — even if edited at resume, it must never get receipt_accuracy_* scores."""
    spy = spy_langfuse()
    respx.post(method_url(f"{API}.create_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "EXP-99"}})
    )
    model = ScriptedChatModel(
        responses=[
            multi_call(("create_expense", {"amount": 20, "category": "Coffee"})),
            answer("Added it."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "add a 20 coffee expense")  # no image

    await _resume(graph, member, {"selected": ["w0"], "edits": {"w0": {"amount": 25}}})

    assert all(t.scores == [] for t in spy.traces)


async def test_non_receipt_photo_asks_instead_of_proposing(member):
    model = ScriptedChatModel(responses=[answer("That doesn't look like a receipt — what's this?")])
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "", image=IMAGE)

    kinds = [k for k, _ in events]
    assert "needs_confirmation" not in kinds and kinds[-1] == "done"
