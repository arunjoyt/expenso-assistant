"""The LangGraph agent — a hand-rolled `agent <-> tools` state graph.

Binds the `tools.py` functions **directly** as LangChain tools (ADR 0008): no
MCP protocol, no `langchain[mcp]`, no MCP client anywhere in this package. The
graph is hand-rolled rather than `create_react_agent` so the per-turn tool-call
cap can live in graph state and a `propose` node can sit in front of the writes.

`tools=` defaults to **all** tools (the interactive turn). A message with any
write call routes to `propose` (P6-S7) — the write functions never run inline;
read-only calls still go to `tools`. Proactive runs pass `tools=READ_TOOLS`, so
`route()` can never reach `propose` there.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .. import tools as tool_defs
from ..config import get_settings
from .prompt import render_system_prompt
from .propose import propose_node


class ToolCapExceeded(RuntimeError):
    """The agent asked for more tool calls in one turn than `run_max_tool_calls`."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_call_count: int
    # Reset fresh in `inputs` at the start of every `/chat` turn (like
    # `tool_call_count`), so it's a pure function of state — unlike the
    # `tools.py` contextvar, which reads back as "assistant" on the resume
    # replay of `propose_node` (P7-S1 grill: the node re-runs from the top on
    # resume, so anything it reads must survive in state, not ambient context).
    entry_method: str


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
        image = (config.get("configurable") or {}).get("receipt_image")
        messages = _with_receipt_image(state["messages"], image)
        reply = await model_with_tools.ainvoke([prompt, *messages], config)
        return {"messages": [reply]}

    async def tools_node(state: AgentState, config) -> dict:
        result = await tool_node.ainvoke(state, config)
        result["tool_call_count"] = state.get("tool_call_count", 0) + 1
        return result

    def route(state: AgentState) -> str:
        calls = getattr(state["messages"][-1], "tool_calls", None)
        if not calls:
            return END
        if any(c["name"] in tool_defs.WRITE_TOOL_NAMES for c in calls):
            return "propose"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("propose", propose_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "propose": "propose", END: END})
    graph.add_edge("tools", "agent")
    graph.add_edge("propose", "agent")
    return graph.compile(checkpointer=checkpointer)


def _resolve_tools(tools: Sequence[Callable[..., Any]] | None) -> list[Callable[..., Any]]:
    return list(tools if tools is not None else tool_defs.ALL_TOOLS)


def _with_receipt_image(messages: list[AnyMessage], image: str | None) -> list[AnyMessage]:
    """A transient copy of `messages` for one model call, with the newest
    `HumanMessage` turned multimodal (P7-S1). Never returned from `agent_node`,
    so the image never reaches checkpointed state — re-built on every loop
    iteration in the turn since each model call is otherwise stateless."""
    if not image:
        return messages
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            text = messages[i].content if isinstance(messages[i].content, str) else ""
            multimodal = HumanMessage(
                content=[
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image}},
                ]
            )
            return [*messages[:i], multimodal, *messages[i + 1 :]]
    return messages
