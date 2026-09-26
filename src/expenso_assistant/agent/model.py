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
    return _chat_openai(settings, settings.openai_model)


def build_fallback_model(settings: Settings) -> BaseChatModel | None:
    """The stock `ModelFallbackMiddleware`'s model (ADR 0010), or None when
    `OPENAI_FALLBACK_MODEL` is unset. Transient errors are already retried by
    the OpenAI SDK itself (2 retries, honouring `Retry-After`), so there is no
    separate model-retry middleware."""
    if not settings.openai_fallback_model:
        return None
    return _chat_openai(settings, settings.openai_fallback_model)


def _chat_openai(settings: Settings, model: str) -> BaseChatModel:
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key or "unset",
        temperature=0,
        streaming=True,
        timeout=settings.http_timeout_seconds,
    )
