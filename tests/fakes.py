"""Test doubles for the agent: a scripted chat model and a LangSmith spy.

The scripted model keeps OpenAI out of every agent test (P6-S5 TEST_PLAN) — it
replays a list of `AIMessage`s, honouring `tool_calls` and `usage_metadata`.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.schemas import Run
from langsmith import Client
from pydantic import Field, PrivateAttr


class ScriptedChatModel(BaseChatModel):
    responses: list = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)
    calls: int = 0
    # Out of the repr: middleware runs trace the model object itself, and a
    # real model holds no prompt (test_agent_observability's image check).
    last_prompt: list = Field(default_factory=list, repr=False)
    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: list, **_: Any) -> ScriptedChatModel:
        self.bound_tools = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _next(self, messages=None):
        self.calls += 1
        if messages is not None:
            _reject_unanswered_tool_calls(messages)
            self.last_prompt = list(messages)
        message = self.responses[self._cursor]
        self._cursor += 1
        return message

    def _generate(self, messages, stop=None, run_manager=None, **_):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **_):
        return self._generate(messages)

    async def _astream(self, messages, stop=None, run_manager=None, **_):
        message = self._next(messages)
        if getattr(message, "tool_calls", None):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_calls=message.tool_calls,
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": json.dumps(call["args"]),
                            "id": call["id"],
                            "index": index,
                        }
                        for index, call in enumerate(message.tool_calls)
                    ],
                )
            )
            return
        words = message.content.split(" ")
        for index, word in enumerate(words):
            last = index == len(words) - 1
            chunk = AIMessageChunk(content=word if last else word + " ")
            if last:
                chunk.usage_metadata = getattr(message, "usage_metadata", None)
            yield ChatGenerationChunk(message=chunk)


class SpyTokenStore:
    """Fake for `observability.PostgresTokenStore` — an in-memory dict keyed by
    user_id (tests run within one 'today', so the date dimension is dropped)."""

    def __init__(self, *, today_tokens: int = 0, get_raises: Exception | None = None):
        self._default = today_tokens
        self._get_raises = get_raises
        self.totals: dict[str, int] = {}
        self.added: list[tuple[str, int]] = []

    def get(self, user_id: str, today) -> int:
        if self._get_raises is not None:
            raise self._get_raises
        return self.totals.get(user_id, self._default)

    def add(self, user_id: str, today, tokens: int) -> None:
        self.added.append((user_id, tokens))
        self.totals[user_id] = self.totals.get(user_id, self._default) + tokens


class SpyTracer(LangChainTracer):
    """The stock tracer, so runs carry the inputs LangSmith would get. Records
    each finished trace's root run, child runs attached."""

    def _persist_run(self, run: Run) -> None:
        self.client.recorded.append(run)


class SpyLangSmith(Client):
    """A LangSmith client that never touches the network: records feedback,
    drops runs, and hands out tracers that record every trace."""

    def __init__(self):
        super().__init__(api_url="http://langsmith.test", api_key="test", auto_batch_tracing=False)
        self.recorded: list[Run] = []
        self.feedback: list[dict] = []

    def create_run(self, *_, **__) -> None:
        pass

    def update_run(self, *_, **__) -> None:
        pass

    def tracer(self) -> SpyTracer:
        return SpyTracer(client=self)

    def create_feedback(self, *, trace_id, key: str, score: float) -> None:
        self.feedback.append({"trace_id": trace_id, "key": key, "score": score})

    def flush(self) -> None:
        pass

    def scores(self, trace: Run) -> dict[str, float]:
        return {f["key"]: f["score"] for f in self.feedback if f["trace_id"] == trace.id}


def _reject_unanswered_tool_calls(messages) -> None:
    """What OpenAI does (HTTP 400): every tool call in the prompt needs a
    `ToolMessage` answer. Without this, a thread left with an orphaned call
    passes every test and fails every turn in prod (ADR 0010)."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(message, AIMessage) and call["id"] not in answered:
                raise ValueError(f"tool call {call['id']} has no ToolMessage")
