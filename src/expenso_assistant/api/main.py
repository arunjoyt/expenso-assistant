"""FastAPI entrypoint — one uvicorn process for the whole service.

P6-S3 ships `/health` and the config-gated `/mcp` mount. The SSE run endpoint,
`/resume` and `/run/proactive` land with the LangGraph agent in P6-S5.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from .. import __version__
from ..config import get_settings
from ..frappe_client import aclose_http


def create_app() -> FastAPI:
    settings = get_settings()
    mcp_app = None
    if settings.mcp_enabled:
        from ..mcp_server import build_mcp_server

        mcp_app = build_mcp_server().http_app(path="/")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async with AsyncExitStack() as stack:
            if mcp_app is not None:
                await stack.enter_async_context(mcp_app.lifespan(mcp_app))
            yield
        await aclose_http()

    app = FastAPI(title="expenso-assistant", version=__version__, lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": __version__, "mcp_enabled": settings.mcp_enabled}

    if mcp_app is not None:
        app.mount("/mcp", mcp_app)

    return app


app = create_app()
