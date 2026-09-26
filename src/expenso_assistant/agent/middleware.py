"""The one custom middleware left (ADR 0010): `ModelCallShaping`.

Everything done to one model call that must never be checkpointed — system
prompt, receipt image, proactive instruction — plus the Insight tag on a
proactive reply. Stock LangChain has no hook for the image or the instruction;
the cap, history editing and write confirmation are all stock (`graph.py`).
"""

from __future__ import annotations

from datetime import UTC, datetime

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from .prompt import render_system_prompt
from .state import RunContext


class ModelCallShaping(AgentMiddleware):
    def __init__(self, *, read_only: bool):
        super().__init__()
        self.read_only = read_only

    async def awrap_model_call(self, request: ModelRequest, handler):
        context: RunContext = request.runtime.context
        messages = with_receipt_image(request.messages, context.receipt_image)
        messages = with_proactive_instruction(messages, context.proactive_instruction)
        prompt = SystemMessage(render_system_prompt(context.today, proactive=self.read_only))
        response = await handler(request.override(system_message=prompt, messages=messages))
        if context.proactive_instruction:
            _tag_as_insight(response.result)
        return response


def with_receipt_image(messages: list[AnyMessage], image: str | None) -> list[AnyMessage]:
    """A copy with the newest `HumanMessage` turned multimodal (P7-S1).
    Re-built on every model call in the turn; never checkpointed."""
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


def with_proactive_instruction(
    messages: list[AnyMessage], instruction: str | None
) -> list[AnyMessage]:
    """A trailing instruction for a proactive run (P7-S2), never checkpointed.
    `agent/proactive.py` decides the job, not the model."""
    if not instruction:
        return messages
    return [*messages, HumanMessage(instruction)]


def _tag_as_insight(messages: list) -> None:
    """Every reply in a proactive turn gets the tag, including a tool-call
    one; `history()` only surfaces the final content-only reply."""
    for message in messages:
        if isinstance(message, AIMessage):
            message.additional_kwargs = {
                **message.additional_kwargs,
                "kind": "insight",
                "posted_at": datetime.now(UTC).isoformat(),
            }
