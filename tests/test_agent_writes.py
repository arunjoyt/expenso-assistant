"""P6-S7 — agent writes: the propose node, the confirm card, /resume.

The scripted model keeps OpenAI out; respx stubs the Frappe REST writes.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from expenso_assistant.agent import session
from expenso_assistant.agent.graph import build_graph
from expenso_assistant.config import get_settings

from .conftest import API, method_url
from .fakes import ScriptedChatModel
from .test_agent_session import answer, collect, run_turn, tool_call

pytestmark = pytest.mark.usefixtures("spy_langfuse_default")

MODIFIED = "2026-03-14 09:00:00"
EXPENSE_ROW = {
    "name": "EXP-17",
    "amount": 4.5,
    "date": "2026-03-14",
    "category": "cat-dining",
    "category_name": "Dining",
    "notes": None,
    "modified": MODIFIED,
}


@pytest.fixture
def spy_langfuse_default(spy_langfuse):
    return spy_langfuse()


def multi_call(*calls: tuple[str, dict]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": a, "id": f"w{i}"} for i, (n, a) in enumerate(calls)],
    )


def _stub_get_expenses(rows: list[dict]) -> None:
    respx.get(method_url(f"{API}.get_expenses")).mock(
        return_value=httpx.Response(200, json={"message": rows})
    )


async def _resume(graph, member, decision) -> list[tuple[str, dict]]:
    return await collect(
        session.resume_turn(graph=graph, member=member, decision=decision, settings=get_settings())
    )


# --- proposing -----------------------------------------------------------


@respx.mock
async def test_write_call_routes_to_propose_and_interrupts(member):
    _stub_get_expenses([EXPENSE_ROW])
    write_route = respx.post(method_url(f"{API}.update_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "EXP-17"}})
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("get_expenses", month=3, year=2026),
            multi_call(("update_expense", {"name": "EXP-17", "amount": 6.0})),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "bump the coffee to 6")

    assert events[-1][0] == "needs_confirmation"
    assert "done" not in [k for k, _ in events]
    (action,) = events[-1][1]["actions"]
    assert action["kind"] == "update"
    assert action["entity"] == "expense"
    assert {"field": "amount", "from": 4.5, "to": 6.0} in action["changes"]
    assert "if_modified_since" not in action  # server-side only
    assert not write_route.called


@respx.mock
async def test_read_only_call_never_reaches_propose(member):
    _stub_get_expenses([EXPENSE_ROW])
    model = ScriptedChatModel(
        responses=[tool_call("get_expenses", month=3, year=2026), answer("You spent 4.5.")]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "march coffee?")

    kinds = [k for k, _ in events]
    assert kinds[-1] == "done" and "needs_confirmation" not in kinds


@respx.mock
async def test_target_not_in_history_nudges_without_interrupting(member):
    _stub_get_expenses([EXPENSE_ROW])
    model = ScriptedChatModel(
        responses=[
            tool_call("get_expenses", month=3, year=2026),
            multi_call(("update_expense", {"name": "EXP-999", "amount": 6.0})),
            answer("I could not find that expense — can you point me to it?"),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "change the other one")

    kinds = [k for k, _ in events]
    assert "needs_confirmation" not in kinds and kinds[-1] == "done"
    assert "re-read" in model.last_prompt[-1].content.lower()


# --- resuming ----------------------------------------------------------


@respx.mock
async def test_confirm_applies_the_selected_actions(member):
    _stub_get_expenses([EXPENSE_ROW])
    update = respx.post(method_url(f"{API}.update_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "EXP-17"}})
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("get_expenses", month=3, year=2026),
            multi_call(("update_expense", {"name": "EXP-17", "amount": 6.0})),
            answer("Updated the coffee to 6."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "bump the coffee to 6")

    events = await _resume(graph, member, {"selected": ["w0"]})

    assert update.called
    body = json.loads(update.calls.last.request.read())
    assert body["amount"] == 6.0
    assert body["if_modified_since"] == MODIFIED
    kinds = [k for k, _ in events]
    assert "token" in kinds and kinds[-1] == "done"


@respx.mock
async def test_confirmed_create_stamps_entry_method_assistant(member):
    create = respx.post(method_url(f"{API}.create_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "EXP-99"}})
    )
    model = ScriptedChatModel(
        responses=[
            multi_call(("create_expense", {"amount": 12, "category": "Groceries"})),
            answer("Added a 12 groceries expense."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "add a 12 groceries expense")

    await _resume(graph, member, {"selected": ["w0"]})

    body = json.loads(create.calls.last.request.read())
    assert body["entry_method"] == "assistant"
    assert "external_message" not in body and "message" not in body


@respx.mock
async def test_deselecting_an_action_writes_only_the_rest(member):
    _stub_get_expenses([EXPENSE_ROW, {**EXPENSE_ROW, "name": "EXP-18"}])
    update = respx.post(method_url(f"{API}.update_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "x"}})
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("get_expenses", month=3, year=2026),
            multi_call(
                ("update_expense", {"name": "EXP-17", "amount": 6.0}),
                ("update_expense", {"name": "EXP-18", "amount": 7.0}),
            ),
            answer("Updated one."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "bump both")

    await _resume(graph, member, {"selected": ["w0"]})

    assert update.call_count == 1
    assert json.loads(update.calls.last.request.read())["name"] == "EXP-17"


@respx.mock
async def test_cancel_writes_nothing(member):
    _stub_get_expenses([EXPENSE_ROW])
    update = respx.post(method_url(f"{API}.update_expense")).mock(
        return_value=httpx.Response(200, json={"message": {}})
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("get_expenses", month=3, year=2026),
            multi_call(("update_expense", {"name": "EXP-17", "amount": 6.0})),
            answer("Okay, left it as is."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "bump the coffee")

    events = await _resume(graph, member, {"selected": []})

    assert not update.called
    assert events[-1][0] == "done"


@respx.mock
async def test_a_per_action_conflict_does_not_abort_the_batch(member):
    _stub_get_expenses([EXPENSE_ROW, {**EXPENSE_ROW, "name": "EXP-18"}])

    def update_side_effect(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        if body["name"] == "EXP-18":
            return httpx.Response(
                417, json={"exc_type": "TimestampMismatchError", "message": "changed"}
            )
        return httpx.Response(200, json={"message": {"name": body["name"]}})

    update = respx.post(method_url(f"{API}.update_expense")).mock(side_effect=update_side_effect)
    model = ScriptedChatModel(
        responses=[
            tool_call("get_expenses", month=3, year=2026),
            multi_call(
                ("update_expense", {"name": "EXP-17", "amount": 6.0}),
                ("update_expense", {"name": "EXP-18", "amount": 7.0}),
            ),
            answer("Updated one; the other changed underneath — take a look."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "bump both")

    events = await _resume(graph, member, {"selected": ["w0", "w1"]})

    assert update.call_count == 2  # both attempted
    assert events[-1][0] == "done"
    assert "changed since you read it" in model.last_prompt[-1].content.lower()


@respx.mock
async def test_partial_confirm_tool_messages_are_row_specific(member):
    coffee = {**EXPENSE_ROW, "name": "EXP-17", "amount": 5, "category_name": "Dining"}
    carrot = {**EXPENSE_ROW, "name": "EXP-18", "amount": 10, "category_name": "Groceries"}
    _stub_get_expenses([coffee, carrot])
    respx.post(method_url(f"{API}.update_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "EXP-18"}})
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("get_expenses", month=3, year=2026),
            multi_call(
                ("update_expense", {"name": "EXP-17", "category": "Other"}),
                ("update_expense", {"name": "EXP-18", "category": "Other"}),
            ),
            answer("Done."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "recategorize the coffee and carrot expenses to Other")

    # Confirm only the carrot row (w1); skip the coffee row (w0).
    await _resume(graph, member, {"selected": ["w1"]})

    tool_messages = {
        m.tool_call_id: m.content for m in model.last_prompt if isinstance(m, ToolMessage)
    }
    # expenso-assistant#5: each outcome must name its own row (Dining vs.
    # Groceries) so the model isn't left to guess which is which from
    # tool_call_id matching alone when it narrates the batch back.
    assert "skipped" in tool_messages["w0"].lower()
    assert "dining" in tool_messages["w0"].lower()
    assert "updated" in tool_messages["w1"].lower()
    assert "groceries" in tool_messages["w1"].lower()


@respx.mock
async def test_batch_cap_limits_the_card(member, monkeypatch):
    monkeypatch.setenv("MAX_PROPOSED_WRITES_PER_TURN", "2")
    get_settings.cache_clear()
    create = respx.post(method_url(f"{API}.create_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "x"}})
    )
    model = ScriptedChatModel(
        responses=[
            multi_call(
                ("create_expense", {"amount": 1}),
                ("create_expense", {"amount": 2}),
                ("create_expense", {"amount": 3}),
            ),
            answer("Added two; one still to go."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "add three expenses")
    assert len(events[-1][1]["actions"]) == 2

    await _resume(graph, member, {"selected": ["w0", "w1"]})
    assert create.call_count == 2
    assert "next" in model.last_prompt[-1].content.lower()


# --- lifecycle -------------------------------------------------------


@respx.mock
async def test_a_new_message_discards_the_pending_card(member):
    _stub_get_expenses([EXPENSE_ROW])
    respx.get(method_url(f"{API}.get_income")).mock(
        return_value=httpx.Response(200, json={"message": []})
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("get_expenses", month=3, year=2026),
            multi_call(("update_expense", {"name": "EXP-17", "amount": 6.0})),
            answer("You have no income in March."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "bump the coffee")

    events = await run_turn(graph, member, "actually, what's my march income?")

    assert events[0] == ("step", {"text": "Discarded the unconfirmed changes"})
    assert events[-1][0] == "done"
    history = await session.history(graph, member, get_settings())
    assert [m["content"] for m in history] == [
        "bump the coffee",
        "actually, what's my march income?",
        "You have no income in March.",
    ]


async def test_resume_with_no_pending_card_is_clean(member):
    graph = build_graph(ScriptedChatModel(responses=[answer("hi")]), checkpointer=InMemorySaver())

    events = await _resume(graph, member, {"selected": []})

    assert events == [("error", {"code": "nothing_to_resume", "message": events[0][1]["message"]})]


@respx.mock
async def test_tool_call_count_persists_across_the_interrupt(member, monkeypatch):
    monkeypatch.setenv("RUN_MAX_TOOL_CALLS", "2")
    get_settings.cache_clear()
    _stub_get_expenses([EXPENSE_ROW])
    respx.post(method_url(f"{API}.update_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "EXP-17"}})
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("get_expenses", month=3, year=2026),  # cycle 1 -> count 1
            multi_call(("update_expense", {"name": "EXP-17", "amount": 6.0})),
            answer("should never be reached"),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "bump it")

    events = await _resume(graph, member, {"selected": ["w0"]})

    assert events[-1][0] == "error" and events[-1][1]["code"] == "tool_cap"


@respx.mock
async def test_resume_opens_its_own_trace_and_skips_the_chat_cap(member, spy_langfuse):
    spy = spy_langfuse()
    _stub_get_expenses([EXPENSE_ROW])
    respx.post(method_url(f"{API}.update_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "EXP-17"}})
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("get_expenses", month=3, year=2026),
            multi_call(("update_expense", {"name": "EXP-17", "amount": 6.0})),
            answer("Updated."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "bump it")
    fetches_after_chat = len(spy.fetch_calls)

    await _resume(graph, member, {"selected": ["w0"]})

    assert len(spy.fetch_calls) == fetches_after_chat  # no daily-cap check on resume
    chat_traces = [t for t in spy.traces if {"feature:chat"} <= set(t.init.get("tags", []))]
    assert len(chat_traces) == 2  # the turn + the resume leg


# --- shapes -----------------------------------------------------------


@respx.mock
async def test_set_budget_is_update_when_a_budget_row_is_in_history(member):
    respx.get(method_url(f"{API}.get_budgets")).mock(
        return_value=httpx.Response(
            200,
            json={
                "message": [{"name": "cat-1", "category_name": "Groceries", "budget_amount": 300}]
            },
        )
    )
    respx.post(method_url(f"{API}.set_budget")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "b1"}})
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("get_budgets", month=3, year=2026),
            multi_call(
                ("set_budget", {"category": "Groceries", "month": 3, "year": 2026, "amount": 400})
            ),
            answer("Budget set."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "raise the groceries budget to 400")

    (action,) = events[-1][1]["actions"]
    assert action["kind"] == "update"
    assert {"field": "amount", "from": 300, "to": 400} in action["changes"]


@respx.mock
async def test_add_category_is_a_create_with_no_stale_guard(member):
    respx.post(method_url(f"{API}.add_category")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "cat-new"}})
    )
    model = ScriptedChatModel(
        responses=[
            multi_call(("add_category", {"name": "Travel"})),
            answer("Added the Travel category."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "add a travel category")

    (action,) = events[-1][1]["actions"]
    assert action["kind"] == "create" and action["entity"] == "category"
    assert action["values"] == {"name": "Travel"}
