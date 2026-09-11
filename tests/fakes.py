"""Test doubles for the agent: a scripted chat model and a Langfuse spy.

The scripted model keeps OpenAI out of every agent test (P6-S5 TEST_PLAN) — it
replays a list of `AIMessage`s, honouring `tool_calls` and `usage_metadata`.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, PrivateAttr


class ScriptedChatModel(BaseChatModel):
    responses: list = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)
    calls: int = 0
    last_prompt: list = Field(default_factory=list)
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


class SpyTrace:
    def __init__(self, **init):
        self.init = init
        self.updates: list[dict] = []
        self.generations: list[dict] = []
        self.scores: list[dict] = []

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)

    def generation(self, **kwargs) -> SpyTrace:
        self.generations.append(kwargs)
        return self

    def score(self, **kwargs) -> None:
        self.scores.append(kwargs)


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


class SpyLangfuse:
    def __init__(
        self,
        *,
        today_trace_count: int = 0,
        today_trace_counts: dict[str, int] | None = None,
        fetch_raises: Exception | None = None,
    ):
        self._today = today_trace_count
        # Per-`feature:<x>` tag override — falls back to `_today` for any tag
        # not listed, so existing single-feature tests are unaffected.
        self._today_by_tag = today_trace_counts or {}
        self._fetch_raises = fetch_raises
        self.traces: list[SpyTrace] = []
        self.fetch_calls: list[dict] = []
        self.flushed = 0

    def trace(self, **kwargs) -> SpyTrace:
        trace = SpyTrace(**kwargs)
        self.traces.append(trace)
        return trace

    def fetch_traces(self, **kwargs):
        self.fetch_calls.append(kwargs)
        if self._fetch_raises is not None:
            raise self._fetch_raises
        tag = (kwargs.get("tags") or [None])[0]
        count = self._today_by_tag.get(tag, self._today)
        return SimpleNamespace(data=[object()] * count)

    def flush(self) -> None:
        self.flushed += 1
