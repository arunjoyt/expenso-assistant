"""LangGraph agent — lands in P6-S5 (#94).

`graph.py` will build the state graph that binds `tools.py`'s READ_TOOLS
(always) and WRITE_TOOLS (interactive turns only; a proposal node raises
`interrupt()` for the confirm card). Postgres checkpointer, Langfuse callback,
per-run caps. Nothing here yet — the FastMCP adapter (P6-S3) does not use it.
"""
