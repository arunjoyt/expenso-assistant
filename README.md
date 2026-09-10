# expenso-assistant

Standalone Assistant service for [Expenso](https://github.com/arunjoyt/expenso).
Runs the LangGraph agent and the FastMCP external-connector adapter in one
uvicorn process, alongside (not inside) the Frappe bench.

Design: `docs/adr/0008-in-app-assistant-architecture.md` in the `expenso` repo.
Streaks are tracked as issues **in the `expenso` repo** (P6-S3, P6-S5, P6-S7,
P7-S1).

## Status

| Streak | State |
|--------|-------|
| P6-S3 — scaffold + `tools.py` + FastMCP adapter + PKCE auth | done |
| P6-S5 — LangGraph agent (read-only) + SSE/`/resume` + Langfuse | done |
| P6-S7 — agent writes + confirm-card flow | not started |
| P7-S1 — receipts (multimodal) | not started |

## Layout

```
src/expenso_assistant/
  config.py         settings + the coupled model/pricing constant + run bounds
  frappe_client.py  thin async Frappe REST client, bearer passthrough
  tools.py          the ONE tool definition (reads + writes) — both consumers bind here
  auth.py           resource-server auth: token introspection, OAuth proxy (PKCE),
                    resolve_member() + the FastAPI dependency + thread-id derivation
  mcp_server.py     FastMCP adapter — registers tools.py, SEP-2322 confirm on writes, MCP_ENABLED-gated
  agent/
    graph.py        hand-rolled agent<->tools StateGraph, binds tools.py directly (no MCP)
    model.py        build_model() — the one place OpenAI is named
    prompt.py       system prompt, today's date filled in per run
    observability.py Langfuse trace + explicit-cost callback + daily-cap query
    session.py      one chat turn: drive the graph, emit SSE, roll back on failure
  api/main.py       FastAPI: /health, /chat (SSE), /resume, /history, /mcp mount
```

## Develop

```bash
uv sync --extra dev
uv run pytest          # no Postgres / OpenAI / Langfuse needed — all faked
uv run ruff check .

# run the service against a local Frappe bench
cp .env.example .env   # set FRAPPE_URL etc.
uv run uvicorn expenso_assistant.api.main:app --reload --port 8080
```

## Deploy

`git pull && docker compose up -d --build` on the VPS. The `app` container runs
the checkpointer's `setup()` (Postgres DDL) on startup, so `assistant` must
exist in Postgres first (the compose `init-multiple-dbs.sh` handles it). `/health`
must be green before the in-app Assistant is switched on. Full runbook,
host-capacity notes and the verification checklist live in `docs/DEPLOYMENT.md`
in the `expenso` repo.
