"""The LangGraph agent (P6-S5, #94).

- `graph.py` — hand-rolled `agent <-> tools` state graph, binds `tools.py`
  functions directly (no MCP). Read-only for P6-S5.
- `model.py` — `build_model()`, the one place OpenAI is named.
- `prompt.py` — the system prompt, today's date filled in per run.
- `observability.py` — Langfuse trace + explicit-cost callback + daily-cap query.
- `session.py` — one chat turn: drive the graph, emit SSE, roll back on failure.
"""
