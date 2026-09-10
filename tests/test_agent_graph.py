"""The graph binds `tools.py` read functions directly — no MCP anywhere."""

from __future__ import annotations

import ast
import pathlib

from langgraph.checkpoint.memory import InMemorySaver

from expenso_assistant import tools
from expenso_assistant.agent.graph import bound_tool_names, build_graph

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
