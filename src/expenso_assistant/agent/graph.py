"""The LangGraph agent — a hand-rolled `agent <-> tools` state graph.

Binds the `tools.py` functions **directly** as LangChain tools (ADR 0008): no
MCP protocol, no `langchain[mcp]`, no MCP client anywhere in this package. The
graph is hand-rolled rather than `create_react_agent` so P6-S7 can slot a
proposal node in front of the write tools and P7-S2 can bind the read-only set,
and so the per-turn tool-call cap can live in graph state.

P6-S5 ships the read-only graph. `tools=` defaults to `READ_TOOLS`; a caller
never gets a write tool here yet.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .. import tools as tool_defs
from ..config import get_settings
from .prompt import render_system_prompt


class ToolCapExceeded(RuntimeError):
    """The agent asked for more tool calls in one turn than `run_max_tool_calls`."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_call_count: int


def bound_tool_names(tools: Sequence[Callable[..., Any]] | None = None) -> set[str]:
    """The tool names a graph built with `tools` would bind. Used by the
    'binds the read tools directly' test and for logging."""
    return {fn.__name__ for fn in _resolve_tools(tools)}


def build_graph(
    model: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver,
    tools: Sequence[Callable[..., Any]] | None = None,
):
    fns = _resolve_tools(tools)
    lc_tools = [
        StructuredTool.from_function(
            coroutine=fn, name=fn.__name__, description=fn.__doc__ or fn.__name__
        )
        for fn in fns
    ]
    model_with_tools = model.bind_tools(lc_tools)
    tool_node = ToolNode(lc_tools)
    max_tool_calls = get_settings().run_max_tool_calls

    async def agent_node(state: AgentState, config) -> dict:
        if state.get("tool_call_count", 0) >= max_tool_calls:
            raise ToolCapExceeded(f"more than {max_tool_calls} tool calls in one turn")
        today = date.fromisoformat(config["configurable"]["today"])
        prompt = SystemMessage(render_system_prompt(today))
        reply = await model_with_tools.ainvoke([prompt, *state["messages"]], config)
        return {"messages": [reply]}

    async def tools_node(state: AgentState, config) -> dict:
        result = await tool_node.ainvoke(state, config)
        result["tool_call_count"] = state.get("tool_call_count", 0) + 1
        return result

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


def _resolve_tools(tools: Sequence[Callable[..., Any]] | None) -> list[Callable[..., Any]]:
    return list(tools if tools is not None else tool_defs.READ_TOOLS)
