"""The graph binds `tools.py` read functions directly — no MCP anywhere."""

from __future__ import annotations

import ast
import pathlib
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver

from expenso_assistant import tools
from expenso_assistant.agent.graph import _as_lc_tool, bound_tool_names, build_graph
from expenso_assistant.agent.state import RunContext

from .fakes import ScriptedChatModel

_AGENT_DIR = pathlib.Path(__file__).parent.parent / "src" / "expenso_assistant" / "agent"


async def test_interactive_graph_binds_read_and_write_tools_directly():
    model = ScriptedChatModel(responses=[AIMessage("hi")])
    await _one_model_call(build_graph(model, checkpointer=InMemorySaver()))

    all_names = {fn.__name__ for fn in tools.ALL_TOOLS}
    assert set(model.bound_tools) == all_names
    assert bound_tool_names() == all_names


async def test_read_only_graph_binds_no_write_tool():
    """Proactive runs (P7-S2) pass tools=READ_TOOLS — no write tool is bound,
    and the human-in-the-loop confirm step is left out of the graph."""
    model = ScriptedChatModel(responses=[AIMessage("hi")])
    graph = build_graph(model, checkpointer=InMemorySaver(), tools=tools.READ_TOOLS)
    await _one_model_call(graph)

    write_names = {fn.__name__ for fn in tools.WRITE_TOOLS}
    assert set(model.bound_tools) == {fn.__name__ for fn in tools.READ_TOOLS}
    assert not set(model.bound_tools) & write_names
    assert not any("HumanInTheLoop" in node for node in graph.get_graph().nodes)


def test_every_tool_argument_reaches_the_model_described():
    """The model sees only the schema. An argument with no `Args:` entry in
    its docstring reaches the model as a bare name and type."""
    for fn in tools.ALL_TOOLS:
        schema = convert_to_openai_tool(_as_lc_tool(fn))["function"]
        for arg, spec in schema["parameters"]["properties"].items():
            assert spec.get("description"), f"{fn.__name__}.{arg} has no description"
        assert "Args:" not in schema["description"]


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


async def _one_model_call(graph) -> None:
    """`create_agent` binds tools per model call, not at build time."""
    await graph.ainvoke(
        {"messages": [HumanMessage("hi")]},
        {"configurable": {"thread_id": "t"}},
        context=RunContext(today=date(2026, 3, 14)),
    )
