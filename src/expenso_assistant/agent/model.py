"""The one place the concrete LLM provider is named.

ADR 0008 keeps LangChain's model abstraction as cheap provider-swap insurance,
not a runtime switch: v1 is OpenAI, and a swap is this function plus the
matching `config.MODEL_PRICING` row in one reviewed commit.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..config import Settings


def build_model(settings: Settings) -> BaseChatModel:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key or "unset",
        temperature=0,
        streaming=True,
        timeout=settings.http_timeout_seconds,
    )
