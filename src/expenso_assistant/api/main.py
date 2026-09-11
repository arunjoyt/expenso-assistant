"""FastAPI entrypoint — one uvicorn process for the whole service.

- `/health`
- `/mcp` — the FastMCP external adapter, mounted only when `MCP_ENABLED`
- `POST /chat` — one agent turn, streamed as SSE
- `POST /resume` — decision on a pending confirm card (endpoint shell in P6-S5;
  the proposal node and real payload land in P6-S7)
- `GET  /history` / `DELETE /history` — the Member's thread
- `POST /run/proactive` — the Frappe scheduler's per-Member trigger (P7-S2);
  fire-and-forget, `202` immediately, the graph runs as a background task

Every Assistant endpoint derives the thread from the token's Member — no client
ever supplies a thread id (P6-S5 grill).
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import __version__
from .. import tools as tool_defs
from ..agent import proactive, session
from ..agent.graph import build_graph
from ..agent.model import build_model
from ..auth import AuthedMember, MemberDep
from ..config import get_settings
from ..frappe_client import aclose_http

_SSE = "text/event-stream"


class ChatIn(BaseModel):
    message: str
    # A `data:image/jpeg;base64,...` data URI (P7-S1) — the frontend always
    # re-encodes to JPEG client-side, so this is the one shape the service
    # needs to accept. Never persisted; see agent/session.py.
    image: str | None = None


class ResumeIn(BaseModel):
    decision: dict = {}


class ProactiveIn(BaseModel):
    job: str


def create_app(*, graph=None, checkpointer=None, read_only_graph=None) -> FastAPI:
    settings = get_settings()
    injected = graph is not None and checkpointer is not None

    mcp_app = None
    if settings.mcp_enabled:
        from ..mcp_server import build_mcp_server

        mcp_app = build_mcp_server().http_app(path="/")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            if mcp_app is not None:
                await stack.enter_async_context(mcp_app.lifespan(mcp_app))
            if injected:
                app.state.graph, app.state.checkpointer = graph, checkpointer
                app.state.read_only_graph = read_only_graph
            else:
                app.state.checkpointer = await _build_checkpointer(stack, settings.database_url)
                model = build_model(settings)
                app.state.graph = build_graph(model, checkpointer=app.state.checkpointer)
                # A separate compiled graph bound to READ_TOOLS only, sharing
                # the same checkpointer/thread — proactive runs (P7-S2) can
                # never produce a write tool-call, a binding-level guarantee.
                app.state.read_only_graph = build_graph(
                    model, checkpointer=app.state.checkpointer, tools=tool_defs.READ_TOOLS
                )
            yield
        await aclose_http()

    app = FastAPI(title="expenso-assistant", version=__version__, lifespan=lifespan)
    if injected:  # tests skip the lifespan; wire state up front
        app.state.graph, app.state.checkpointer = graph, checkpointer
        app.state.read_only_graph = read_only_graph

    # The in-app Assistant tab calls /chat from the Frappe origin (P6-S6). The
    # bearer rides an Authorization header, so that header must be allowed.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": __version__, "mcp_enabled": settings.mcp_enabled}

    @app.post("/chat")
    async def chat(body: ChatIn, request: Request, member: AuthedMember = MemberDep):
        if len(body.message) > settings.max_chat_message_chars:
            raise HTTPException(413, "message is too long")
        if body.image and len(body.image) > settings.max_receipt_image_chars:
            raise HTTPException(413, "image is too large")
        stream = session.stream_turn(
            graph=request.app.state.graph,
            member=member,
            text=body.message,
            settings=settings,
            image=body.image,
        )
        return StreamingResponse(stream, media_type=_SSE)

    @app.post("/resume")
    async def resume(body: ResumeIn, request: Request, member: AuthedMember = MemberDep):
        stream = session.resume_turn(
            graph=request.app.state.graph, member=member, decision=body.decision, settings=settings
        )
        return StreamingResponse(stream, media_type=_SSE)

    @app.get("/history")
    async def get_history(request: Request, member: AuthedMember = MemberDep) -> dict:
        return {"messages": await session.history(request.app.state.graph, member, settings)}

    @app.delete("/history")
    async def clear_history(request: Request, member: AuthedMember = MemberDep) -> dict:
        await session.clear_thread(request.app.state.checkpointer, member)
        return {"status": "cleared"}

    @app.post("/run/proactive", status_code=status.HTTP_202_ACCEPTED)
    async def run_proactive(
        body: ProactiveIn,
        background_tasks: BackgroundTasks,
        request: Request,
        member: AuthedMember = MemberDep,
    ) -> dict:
        background_tasks.add_task(
            proactive.run_proactive_job,
            job=body.job,
            member=member,
            graph=request.app.state.read_only_graph,
            settings=settings,
        )
        return {"status": "accepted"}

    if mcp_app is not None:
        app.mount("/mcp", mcp_app)

    return app


async def _build_checkpointer(stack: AsyncExitStack, database_url: str):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    checkpointer = await stack.enter_async_context(
        AsyncPostgresSaver.from_conn_string(database_url)
    )
    await checkpointer.setup()
    return checkpointer


app = create_app()
