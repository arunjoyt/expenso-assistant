"""What the agent persists per thread, and what it gets per run.

`ExpensoState` is checkpointed in Postgres. `RunContext` is not: it is passed
fresh on every `/chat`, `/resume` and proactive run (ADR 0010), so a receipt
image or a proactive instruction can never leak into stored history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import NotRequired

from langchain.agents.middleware import AgentState


class ExpensoState(AgentState):
    # State, not context: an approved write runs on the `/resume` leg, which
    # has no receipt, so the turn's original value must persist until then.
    entry_method: NotRequired[str]


@dataclass(frozen=True)
class RunContext:
    today: date
    receipt_image: str | None = None
    proactive_instruction: str | None = None
