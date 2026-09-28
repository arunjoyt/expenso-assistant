"""The LangGraph agent — LangChain v1 `create_agent` with stock middleware.

Binds the `tools.py` functions **directly** as LangChain tools (ADR 0008): no
MCP protocol, no `langchain[mcp]`, no MCP client anywhere in this package.

ADR 0010: the loop, the tool-call cap, history editing and the write
confirmation are all stock LangChain. The one custom middleware is
`ModelCallShaping` (prompt, receipt image, proactive instruction).

`tools=` defaults to **all** tools (the interactive turn): every write tool
pauses on the stock human-in-the-loop confirm card. Proactive runs pass
`tools=READ_TOOLS`, which leaves the confirm step out entirely.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ModelFallbackMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from .. import tools as tool_defs
from ..config import get_settings
from ..frappe_client import FrappeError
from .describe import describe_write
from .middleware import ModelCallShaping
from .state import ExpensoState, RunContext

__all__ = ["bound_tool_names", "build_graph"]


def build_graph(
    model: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver,
    tools: Sequence[Callable[..., Any]] | None = None,
    fallback_model: BaseChatModel | None = None,
):
    fns = _resolve_tools(tools)
    settings = get_settings()
    # No write tool bound means a proactive run (P7-S2): picks the prompt's tone.
    read_only = not any(fn.__name__ in tool_defs.WRITE_TOOL_NAMES for fn in fns)

    # `wrap_*` hooks nest in list order: the first is the outermost.
    middleware = [
        ModelCallShaping(read_only=read_only),
        *([ModelFallbackMiddleware(fallback_model)] if fallback_model else []),
        # Transient: clears old tool outputs from what the model sees; the
        # checkpointed thread stays whole (the GLOSSARY promise).
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=settings.chat_history_token_budget)]
        ),
    ]
    if not read_only:
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=_confirm_every_write()))
    # A Frappe rejection (validation, a stale-write conflict) goes back to the
    # model as an error ToolMessage, so the other calls in a batch still run.
    middleware.append(ToolErrorMiddleware(on_error=_frappe_error_text))
    # Inside ToolErrorMiddleware, as its docs require. Reads only: a write that
    # timed out may still have committed, so retrying it could duplicate a row.
    middleware.append(
        ToolRetryMiddleware(
            tools=[fn.__name__ for fn in fns if fn.__name__ not in tool_defs.WRITE_TOOL_NAMES],
            retry_on=_is_transient,
            max_retries=2,
            initial_delay=0.5,
            on_failure="error",
        )
    )
    # Last, so its `after_model` hook runs first: a turn over the cap fails
    # before a confirm card is shown for it.
    middleware.append(
        ToolCallLimitMiddleware(run_limit=settings.run_max_tool_calls, exit_behavior="error")
    )

    return create_agent(
        model,
        tools=[_as_lc_tool(fn) for fn in fns],
        middleware=middleware,
        state_schema=ExpensoState,
        context_schema=RunContext,
        checkpointer=checkpointer,
    )


def bound_tool_names(tools: Sequence[Callable[..., Any]] | None = None) -> set[str]:
    """The tool names a graph built with `tools` would bind. Used by the
    'binds the read tools directly' test and for logging."""
    return {fn.__name__ for fn in _resolve_tools(tools)}


def _confirm_every_write() -> dict[str, InterruptOnConfig]:
    config = InterruptOnConfig(
        allowed_decisions=["approve", "edit", "reject"], description=describe_write
    )
    return {name: config for name in tool_defs.WRITE_TOOL_NAMES}


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, httpx.TransportError):  # connect / read timeout, reset
        return True
    return isinstance(exc, FrappeError) and exc.status_code in _TRANSIENT_STATUS


_TRANSIENT_STATUS = frozenset({429, 502, 503, 504})


def _frappe_error_text(exc: Exception, request) -> str | None:
    if not isinstance(exc, FrappeError):
        return None  # anything else is a bug: let it fail the turn
    name = request.tool_call["name"]
    if exc.exc_type == "TimestampMismatchError":
        return f"{name}: the row changed since you read it — re-read it and propose again."
    # Only the Member-facing text: the raw body can carry a traceback.
    return f"{name} could not be applied: {exc.user_message}"


def _as_lc_tool(fn: Callable[..., Any]) -> StructuredTool:
    # `Args:` in the docstring becomes each argument's schema description.
    return StructuredTool.from_function(coroutine=fn, name=fn.__name__, parse_docstring=True)


def _resolve_tools(tools: Sequence[Callable[..., Any]] | None) -> list[Callable[..., Any]]:
    return list(tools if tools is not None else tool_defs.ALL_TOOLS)
