"""Agent writes: the stock human-in-the-loop confirm card and /resume (ADR 0010).

The scripted model keeps OpenAI out; respx stubs the Frappe REST writes.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END

from expenso_assistant.agent import session
from expenso_assistant.agent.graph import build_graph
from expenso_assistant.config import get_settings

from .conftest import API, method_url
from .fakes import ScriptedChatModel
from .test_agent_session import answer, collect, run_turn, tool_call

pytestmark = pytest.mark.usefixtures("spy_tracing_default")

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
def spy_tracing_default(spy_tracing):
    return spy_tracing()


def multi_call(*calls: tuple[str, dict]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": a, "id": f"w{i}"} for i, (n, a) in enumerate(calls)],
    )


def _stub_get_expenses(rows: list[dict]):
    return respx.get(method_url(f"{API}.get_expenses")).mock(
        return_value=httpx.Response(200, json={"message": rows})
    )


def approve() -> dict:
    return {"type": "approve"}


def reject() -> dict:
    return {"type": "reject"}


def edit(name: str, args: dict) -> dict:
    return {"type": "edit", "edited_action": {"name": name, "args": args}}


async def _resume(graph, member, decision) -> list[tuple[str, dict]]:
    return await collect(
        session.resume_turn(graph=graph, member=member, decision=decision, settings=get_settings())
    )


# --- proposing -----------------------------------------------------------


@respx.mock
async def test_write_call_pauses_on_a_confirm_card(member):
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
    (request,) = events[-1][1]["action_requests"]
    assert request["name"] == "update_expense"
    assert request["args"] == {"name": "EXP-17", "amount": 6.0}
    assert "4.5 · Dining" in request["description"]
    assert "amount: 4.5 → 6.0" in request["description"]
    (review,) = events[-1][1]["review_configs"]
    assert review["allowed_decisions"] == ["approve", "edit", "reject"]
    assert not write_route.called


@respx.mock
async def test_read_only_call_never_pauses(member):
    _stub_get_expenses([EXPENSE_ROW])
    model = ScriptedChatModel(
        responses=[tool_call("get_expenses", month=3, year=2026), answer("You spent 4.5.")]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "march coffee?")

    kinds = [k for k, _ in events]
    assert kinds[-1] == "done" and "needs_confirmation" not in kinds


@respx.mock
async def test_a_target_not_in_history_is_flagged_on_the_card(member):
    """ADR 0010 dropped the 're-read first' nudge: the card still opens, and
    says plainly that the row was not read."""
    _stub_get_expenses([EXPENSE_ROW])
    model = ScriptedChatModel(
        responses=[
            tool_call("get_expenses", month=3, year=2026),
            multi_call(("update_expense", {"name": "EXP-999", "amount": 6.0})),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "change the other one")

    (request,) = events[-1][1]["action_requests"]
    assert "EXP-999 (not in what you've read)" in request["description"]


@respx.mock
async def test_reads_in_a_write_message_run_while_the_writes_wait(member):
    """ADR 0010 dropped the mixed-message bounce: reads run, writes pause."""
    reads = _stub_get_expenses([EXPENSE_ROW])
    model = ScriptedChatModel(
        responses=[
            multi_call(
                ("get_expenses", {"month": 3, "year": 2026}),
                ("create_expense", {"amount": 2}),
            ),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "read and add")

    assert events[-1][0] == "needs_confirmation"
    assert [r["name"] for r in events[-1][1]["action_requests"]] == ["create_expense"]
    assert not reads.called  # runs only once the card is decided


# --- resuming ----------------------------------------------------------


@respx.mock
async def test_approve_runs_the_write(member):
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

    events = await _resume(graph, member, {"decisions": [approve()]})

    assert update.called
    body = json.loads(update.calls.last.request.read())
    assert body["amount"] == 6.0
    kinds = [k for k, _ in events]
    assert "token" in kinds and kinds[-1] == "done"


@respx.mock
async def test_the_stale_write_guard_is_only_what_the_model_passes(member):
    """Known issue (ADR 0010, expenso-assistant#9): the service no longer injects the row's
    `modified` value. The guard holds only if the model passes it."""
    _stub_get_expenses([EXPENSE_ROW])
    update = respx.post(method_url(f"{API}.update_expense")).mock(
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
    await run_turn(graph, member, "bump the coffee to 6")

    await _resume(graph, member, {"decisions": [approve()]})

    assert json.loads(update.calls.last.request.read()).get("if_modified_since") is None


@respx.mock
async def test_approved_create_stamps_entry_method_assistant(member):
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

    await _resume(graph, member, {"decisions": [approve()]})

    body = json.loads(create.calls.last.request.read())
    assert body["entry_method"] == "assistant"
    assert "external_message" not in body and "message" not in body


@respx.mock
async def test_rejecting_one_action_writes_only_the_rest(member):
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

    await _resume(graph, member, {"decisions": [approve(), reject()]})

    assert update.call_count == 1
    assert json.loads(update.calls.last.request.read())["name"] == "EXP-17"
    rejected = next(m for m in model.last_prompt if getattr(m, "tool_call_id", None) == "w1")
    assert "rejected" in rejected.content.lower()


@respx.mock
async def test_rejecting_everything_writes_nothing(member):
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

    events = await _resume(graph, member, {"decisions": [reject()]})

    assert not update.called
    assert events[-1][0] == "done"


@respx.mock
async def test_an_edit_runs_the_edited_args_and_tells_the_model(member):
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
    await run_turn(graph, member, "add 12 groceries")

    await _resume(
        graph,
        member,
        {"decisions": [edit("create_expense", {"amount": 15, "category": "Groceries"})]},
    )

    assert json.loads(create.calls.last.request.read())["amount"] == 15
    result = next(m for m in model.last_prompt if isinstance(m, ToolMessage))
    assert '"amount": 15' in result.content  # the model is told what actually ran


@respx.mock
async def test_a_frappe_conflict_on_one_write_does_not_stop_the_others(member):
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

    events = await _resume(graph, member, {"decisions": [approve(), approve()]})

    assert update.call_count == 2  # both attempted
    assert events[-1][0] == "done"
    conflict = next(m for m in model.last_prompt if getattr(m, "tool_call_id", None) == "w1")
    assert "changed since you read it" in conflict.content.lower()


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


@respx.mock
async def test_a_card_whose_paused_task_was_lost_is_still_discarded(member):
    """A deploy that renames graph nodes (ADR 0010) drops the paused task but
    keeps the unanswered write call in the thread. It must still be discarded,
    or OpenAI rejects every later turn."""
    _stub_get_expenses([EXPENSE_ROW])
    model = ScriptedChatModel(
        responses=[
            tool_call("get_expenses", month=3, year=2026),
            multi_call(("update_expense", {"name": "EXP-17", "amount": 6.0})),
            answer("Okay, left it as is."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "bump the coffee")
    config = {"configurable": {"thread_id": member.thread_id}}
    await graph.aupdate_state(config, None, as_node=END)  # the lost task
    assert await session.pending_card(graph, config) is None

    events = await run_turn(graph, member, "never mind")

    assert events[0] == ("step", {"text": "Discarded the unconfirmed changes"})
    assert events[-1][0] == "done"
    history = await session.history(graph, member, get_settings())
    assert [m["content"] for m in history][-1] == "Okay, left it as is."


async def test_resume_with_no_pending_card_is_clean(member):
    graph = build_graph(ScriptedChatModel(responses=[answer("hi")]), checkpointer=InMemorySaver())

    events = await _resume(graph, member, {"decisions": []})

    assert events == [("error", {"code": "nothing_to_resume", "message": events[0][1]["message"]})]


@respx.mock
async def test_the_tool_call_cap_resets_on_the_resume_leg(member, monkeypatch):
    """ADR 0010: the stock per-run cap resets when /resume starts a new run
    (the old step count spanned the pause). Two calls before the card and
    one after would trip a cap of 2 if it spanned the pause; it does not."""
    monkeypatch.setenv("RUN_MAX_TOOL_CALLS", "2")
    get_settings.cache_clear()
    _stub_get_expenses([EXPENSE_ROW])
    respx.post(method_url(f"{API}.update_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "EXP-17"}})
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("get_expenses", month=3, year=2026),
            multi_call(("update_expense", {"name": "EXP-17", "amount": 6.0})),
            tool_call("get_expenses", month=3, year=2026),
            answer("Updated, and it reads back as 6."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "bump it")

    events = await _resume(graph, member, {"decisions": [approve()]})

    assert events[-1][0] == "done"


@respx.mock
async def test_resume_opens_its_own_trace_and_skips_the_chat_cap(
    member, spy_tracing, spy_token_store
):
    spy = spy_tracing()
    store = spy_token_store()
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
    store.get = _fail_if_called  # no daily-cap check on resume

    await _resume(graph, member, {"decisions": [approve()]})

    chat_traces = [t for t in spy.recorded if "feature:chat" in t.tags]
    assert len(chat_traces) == 2  # the turn + the resume leg


def _fail_if_called(*_):
    raise AssertionError("the daily cap was checked")


# --- card text --------------------------------------------------------


@respx.mock
async def test_set_budget_shows_the_current_amount_when_a_budget_row_was_read(member):
    respx.get(method_url(f"{API}.get_budgets")).mock(
        return_value=httpx.Response(
            200,
            json={
                "message": [{"name": "cat-1", "category_name": "Groceries", "budget_amount": 300}]
            },
        )
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("get_budgets", month=3, year=2026),
            multi_call(
                ("set_budget", {"category": "Groceries", "month": 3, "year": 2026, "amount": 400})
            ),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "raise the groceries budget to 400")

    (request,) = events[-1][1]["action_requests"]
    assert request["description"] == "Budget — Groceries, 3/2026: amount 300 → 400"


@respx.mock
async def test_add_category_reads_as_an_add(member):
    model = ScriptedChatModel(responses=[multi_call(("add_category", {"name": "Travel"}))])
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "add a travel category")

    (request,) = events[-1][1]["action_requests"]
    assert request["description"] == "Add category 'Travel'"
    assert request["args"] == {"name": "Travel"}
