"""The LangGraph agent (P6-S5, #94).

- `graph.py` — `create_agent` with stock middleware (ADR 0010): tool-call
  cap, history editing, human-in-the-loop confirm card. Binds `tools.py`
  functions directly (no MCP).
- `middleware.py` — the one custom middleware: prompt, receipt image,
  proactive instruction.
- `describe.py` — the confirm card's text for each proposed write.
- `state.py` — checkpointed state and the per-run `RunContext`.
- `model.py` — `build_model()`, the one place OpenAI is named.
- `prompt.py` — the system prompt, today's date filled in per run.
- `observability.py` — Langfuse trace + explicit-cost callback + daily-cap query.
- `session.py` — one chat turn: drive the graph, emit SSE, roll back on failure.
"""
