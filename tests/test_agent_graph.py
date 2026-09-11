"""The graph binds `tools.py` read functions directly — no MCP anywhere."""

from __future__ import annotations

import ast
import pathlib

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from expenso_assistant import tools
from expenso_assistant.agent.graph import _token_counter, _windowed, bound_tool_names, build_graph

from .fakes import ScriptedChatModel

_AGENT_DIR = pathlib.Path(__file__).parent.parent / "src" / "expenso_assistant" / "agent"


def test_interactive_graph_binds_read_and_write_tools_directly():
    model = ScriptedChatModel()
    build_graph(model, checkpointer=InMemorySaver())

    all_names = {fn.__name__ for fn in tools.ALL_TOOLS}
    assert set(model.bound_tools) == all_names
    assert bound_tool_names() == all_names


def test_read_only_graph_binds_no_write_tool():
    """Proactive runs (P7-S2) pass tools=READ_TOOLS — no write tool is bound, so
    route() can never reach the propose node there."""
    model = ScriptedChatModel()
    build_graph(model, checkpointer=InMemorySaver(), tools=tools.READ_TOOLS)

    write_names = {fn.__name__ for fn in tools.WRITE_TOOLS}
    assert not set(model.bound_tools) & write_names


def test_windowed_keeps_only_the_most_recent_messages_under_budget():
    """2026-09-11 update: a transient trim, not a checkpoint prune — this
    tests the pure function, not the checkpoint (see test_agent_session.py
    for proof the persisted thread is untouched)."""
    count = _token_counter("gpt-4o-mini")
    messages = [HumanMessage(f"message number {i} padded with extra words") for i in range(20)]

    trimmed = _windowed(messages, count, budget=50)

    assert trimmed[-1].content == messages[-1].content
    assert messages[0].content not in [m.content for m in trimmed]
    assert count(trimmed) <= 50


def test_windowed_is_a_noop_under_budget():
    messages = [HumanMessage("hi")]
    count = _token_counter("gpt-4o-mini")

    assert _windowed(messages, count, budget=10_000) == messages


def test_agent_package_imports_no_mcp():
    """No MCP protocol in the agent's path (ADR 0008) — checked against real
    import statements, not prose."""
    for path in _AGENT_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert name != "mcp"
                assert not name.startswith(("mcp.", "langchain_mcp", "fastmcp"))
