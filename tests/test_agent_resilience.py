"""ADR 0010: stock retry and fallback middleware.

Reads that hit a transient Frappe error are retried; writes never are (a
write that timed out may have committed). A failing primary model falls back
to `OPENAI_FALLBACK_MODEL`'s model when one is set.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from expenso_assistant.agent.graph import build_graph
from expenso_assistant.frappe_client import FrappeError

from .conftest import API, method_url
from .fakes import ScriptedChatModel
from .test_agent_session import answer, run_turn, tool_call
from .test_agent_writes import _resume, approve, multi_call

pytestmark = pytest.mark.usefixtures("spy_tracing_default")


@pytest.fixture
def spy_tracing_default(spy_tracing):
    return spy_tracing()


class BrokenChatModel(BaseChatModel):
    """A primary model whose provider is down."""

    @property
    def _llm_type(self) -> str:
        return "broken"

    def bind_tools(self, tools, **_):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **_):
        raise RuntimeError("provider down")

    async def _agenerate(self, messages, stop=None, run_manager=None, **_):
        raise RuntimeError("provider down")


@respx.mock
async def test_a_transient_read_error_is_retried(member):
    reads = respx.get(method_url(f"{API}.get_expenses")).mock(
        side_effect=[
            httpx.Response(503, text="Service Unavailable"),
            httpx.Response(200, json={"message": []}),
        ]
    )
    model = ScriptedChatModel(
        responses=[tool_call("get_expenses", month=3, year=2026), answer("No expenses.")]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "march?")

    assert reads.call_count == 2
    assert events[-1][0] == "done"


@respx.mock
async def test_a_failing_write_is_never_retried(member):
    create = respx.post(method_url(f"{API}.create_expense")).mock(
        return_value=httpx.Response(503, text="Service Unavailable")
    )
    model = ScriptedChatModel(
        responses=[
            multi_call(("create_expense", {"amount": 2})),
            answer("That did not go through — try again in a moment."),
        ]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "add 2")

    events = await _resume(graph, member, {"decisions": [approve()]})

    assert create.call_count == 1
    assert events[-1][0] == "done"  # the error went back to the model, not the turn
    assert "could not be applied" in model.last_prompt[-1].content


@respx.mock
async def test_the_model_sees_the_frappe_message_but_never_the_traceback(member):
    """A Frappe v1 error body as a system user with tracebacks allowed gets it."""
    body = {
        "exc_type": "ValidationError",
        "exception": "frappe.exceptions.ValidationError: Amount must be positive",
        "exc": json.dumps(['Traceback (most recent call last):\n  File "/srv/api.py"']),
        "_server_messages": json.dumps(
            [json.dumps({"message": "Amount must be <b>positive</b>", "indicator": "red"})]
        ),
    }
    respx.post(method_url(f"{API}.create_expense")).mock(
        return_value=httpx.Response(417, json=body)
    )
    model = ScriptedChatModel(
        responses=[multi_call(("create_expense", {"amount": -2})), answer("It was rejected.")]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    await run_turn(graph, member, "add -2")

    await _resume(graph, member, {"decisions": [approve()]})

    seen = model.last_prompt[-1].content
    assert seen == "create_expense could not be applied: Amount must be positive"


def test_a_non_json_frappe_error_reads_as_its_status():
    error = FrappeError(502, "<html><body>Bad Gateway</body></html>")

    assert (error.exc_type, error.user_message) == (None, "HTTP 502")


async def test_a_failing_model_falls_back(member):
    fallback = ScriptedChatModel(responses=[answer("Answered by the fallback.")])
    graph = build_graph(BrokenChatModel(), checkpointer=InMemorySaver(), fallback_model=fallback)

    events = await run_turn(graph, member, "hi")

    assert events[-1][0] == "done"
    assert fallback.calls == 1


async def test_with_no_fallback_a_failing_model_fails_the_turn(member):
    graph = build_graph(BrokenChatModel(), checkpointer=InMemorySaver())

    events = await run_turn(graph, member, "hi")

    assert events[-1] == ("error", {"code": "internal", "message": events[-1][1]["message"]})
