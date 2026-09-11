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
from datetime import UTC, date, datetime
from functools import lru_cache
from typing import Annotated, Any, TypedDict

import tiktoken
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, trim_messages
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
    history_token_budget = get_settings().chat_history_token_budget
    count_tokens = _token_counter(get_settings().openai_model)
    # No write tool in `fns` means this is a proactive run (P7-S2) — used to pick
    # the system prompt's tone, not a re-check of what's bound (route() already
    # enforces that structurally).
    read_only = not any(fn.__name__ in tool_defs.WRITE_TOOL_NAMES for fn in fns)

    async def agent_node(state: AgentState, config) -> dict:
        if state.get("tool_call_count", 0) >= max_tool_calls:
            raise ToolCapExceeded(f"more than {max_tool_calls} tool calls in one turn")
        today = date.fromisoformat(config["configurable"]["today"])
        configurable = config.get("configurable") or {}
        instruction = configurable.get("proactive_instruction")
        prompt = SystemMessage(render_system_prompt(today, proactive=read_only))
        messages = _windowed(state["messages"], count_tokens, history_token_budget)
        messages = _with_receipt_image(messages, configurable.get("receipt_image"))
        messages = _with_proactive_instruction(messages, instruction)
        reply = await model_with_tools.ainvoke([prompt, *messages], config)
        if instruction:
            # Tags this run's reply(ies) as an Insight for the transcript
            # (P7-S2) — every reply in a proactive turn gets it, including an
            # intermediate tool-call message, but `history()` only surfaces the
            # final content-only one, so that's harmless.
            reply.additional_kwargs = {
                **reply.additional_kwargs,
                "kind": "insight",
                "posted_at": datetime.now(UTC).isoformat(),
            }
        return {"messages": [reply]}

    async def tools_node(state: AgentState, config) -> dict:
        result = await tool_node.ainvoke(state, config)
        result["tool_call_count"] = state.get("tool_call_count", 0) + 1
        return result

    def route(state: AgentState) -> str:
        messages = state["messages"]
        if not messages:
            # A proactive thread's state can be emptied by `discard_pending`
            # clearing a stale interactive proposal (no persisted HumanMessage
            # ever anchors it) — expenso-assistant#1. Nothing to route on.
            return END
        calls = getattr(messages[-1], "tool_calls", None)
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


@lru_cache(maxsize=4)
def _encoding_for(model_name: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _token_counter(model_name: str) -> Callable[[list[AnyMessage]], int]:
    """A tiktoken-based counter for `_windowed`, deliberately not routed
    through the bound model's own `get_num_tokens_from_messages`: that works
    for the real `ChatOpenAI` (tiktoken-backed) but not for a plain
    `BaseChatModel` double like the test fakes, which fall through to
    LangChain's default tokenizer and require the `transformers` package.
    Approximate (content only, no per-message role/name overhead) — fine for
    a soft budget, not a billing figure."""
    encoding = _encoding_for(model_name)

    def count(messages: list[AnyMessage]) -> int:
        total = 0
        for message in messages:
            content = message.content
            text = content if isinstance(content, str) else str(content)
            total += len(encoding.encode(text))
        return total

    return count


def _windowed(
    messages: list[AnyMessage], count_tokens: Callable[[list[AnyMessage]], int], budget: int
) -> list[AnyMessage]:
    """A transient token-count trim of persisted history (2026-09-11 grill) —
    the same shape as `_with_receipt_image`/`_with_proactive_instruction`
    below: computed fresh before each model call, never returned from
    `agent_node`, so the checkpoint (and everything `GET /history` replays)
    stays the full, ever-growing thread the GLOSSARY promises — only what's
    sent to the model is bounded. `start_on=("human", "ai")` keeps the window
    from starting mid a tool-call/tool-result pair, which OpenAI's API
    rejects — a `ToolMessage` is never a valid start. It also fixes a real bug
    (expenso-assistant#1): a proactive run's `state["messages"]` never carries
    a persisted `HumanMessage` (the instruction lives in `config`, injected by
    `_with_proactive_instruction` after this runs), so after the first
    tool-call cycle the window was `[AIMessage(tool_call), ToolMessage(...)]`
    with no `human` anchor at all — `start_on="human"` alone returned `[]`,
    silently dropping the tool result every cycle and looping until
    `ToolCapExceeded`. Allowing an `ai` start lets the window begin at that
    `AIMessage` instead."""
    return trim_messages(
        messages,
        max_tokens=budget,
        token_counter=count_tokens,
        strategy="last",
        start_on=("human", "ai"),
    )


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


def _with_proactive_instruction(
    messages: list[AnyMessage], instruction: str | None
) -> list[AnyMessage]:
    """An ephemeral trailing instruction for a proactive run (P7-S2), riding in
    `config` exactly like the receipt image above — never returned from
    `agent_node`, so it is never checkpointed. The job description (which
    Categories crossed budget, which month to summarize) is decided in
    `agent/proactive.py`, not by the model."""
    if not instruction:
        return messages
    return [*messages, HumanMessage(instruction)]
