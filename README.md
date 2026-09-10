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
| P6-S3 — scaffold + `tools.py` + FastMCP adapter + PKCE auth | in progress |
| P6-S5 — LangGraph agent (read-only) + SSE/`/resume` + Langfuse | not started |
| P6-S7 — agent writes + confirm-card flow | not started |
| P7-S1 — receipts (multimodal) | not started |

## Layout

```
src/expenso_assistant/
  config.py         settings + the coupled model/pricing constant
  frappe_client.py  thin async Frappe REST client, bearer passthrough
  tools.py          the ONE tool definition (reads + writes) — both consumers bind here
  auth.py           resource-server auth: Frappe token introspection + OAuth proxy (PKCE)
  mcp_server.py     FastMCP adapter — registers tools.py, SEP-2322 confirm on writes, MCP_ENABLED-gated
  api/main.py       FastAPI: /health, /mcp mount  (SSE run + /resume come in P6-S5)
  agent/            LangGraph agent  (P6-S5)
```

## Develop

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .

# run the service against a local Frappe bench
cp .env.example .env   # set FRAPPE_URL etc.
uv run uvicorn expenso_assistant.api.main:app --reload --port 8080
```

## Deploy

`git pull && docker compose up -d --build` on the VPS. `/health` must be green
before the Frappe-side cutover (P6-S4) deletes `expenso/mcp.py`. Full runbook,
host-capacity notes and the verification checklist live in
`docs/DEPLOYMENT.md` in the `expenso` repo.
