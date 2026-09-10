"""The write-confirmation step (P6-S7).

A message with any write tool-call routes here instead of to `tools`. This node:

1. builds a batched confirm card from the write calls plus the target rows'
   current values already in the tool history (no re-read),
2. raises `interrupt(card)` — the `tools.py` write functions never run inline,
3. on `Command(resume={"selected": [...]})`, calls the write functions for the
   member's selected subset, one `ToolMessage` per original call, applying the
   rest of a batch even when one action hits a concurrency conflict.

`interrupt()` re-runs the node from the top on resume, so nothing here mutates
before that call — the card is built from state, which is pure.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import interrupt

from .. import tools as tool_defs
from ..config import get_settings
from ..frappe_client import FrappeError

_TIMESTAMP_MISMATCH = "TimestampMismatchError"

_READ_FOR_ENTITY = {"expense": "get_expenses", "income": "get_income"}
_LABEL_FIELD = {"expense": "category", "income": "source"}


async def propose_node(state: dict, config) -> dict:
    calls = state["messages"][-1].tool_calls
    bump = {"tool_call_count": state.get("tool_call_count", 0) + 1}

    # The prompt says: read first, then propose writes in a message with no
    # reads. A mixed message is bounced whole rather than partly executed.
    if any(c["name"] not in tool_defs.WRITE_TOOL_NAMES for c in calls):
        return {
            "messages": [
                _tool_msg(
                    c, "Propose writes in a message with no read calls — read in an earlier step."
                )
                for c in calls
            ],
            **bump,
        }

    actions = [_build_action(c, state["messages"]) for c in calls]

    unresolved = [a for a in actions if a.get("unresolved")]
    if unresolved:
        missing = ", ".join(a["summary"] for a in unresolved)
        return {
            "messages": [
                _tool_msg(
                    {"id": a["id"], "name": a["tool"]},
                    f"Not proposed — re-read {missing} first, then propose the whole set again.",
                )
                for a in actions
            ],
            **bump,
        }

    cap = get_settings().max_proposed_writes_per_turn
    shown, overflow = actions[:cap], actions[cap:]

    decision = interrupt({"actions": [_public(a) for a in shown]})
    selected = set((decision or {}).get("selected") or [])

    out = [await _apply(a, a["id"] in selected) for a in shown]
    for a in overflow:
        out.append(
            _tool_msg(
                {"id": a["id"], "name": a["tool"]},
                f"Deferred — over the {cap}-changes-per-card limit. Propose this one again next.",
            )
        )
    return {"messages": out, **bump}


# --- building the card ----------------------------------------------------


def _build_action(call: dict, messages: list) -> dict:
    name = call["name"]
    kind, entity = tool_defs.write_kind(name)
    args = {k: v for k, v in call["args"].items() if v is not None}
    action: dict[str, Any] = {"id": call["id"], "tool": name, "call_args": dict(args)}

    if name in ("create_expense", "create_income", "add_category", "add_source"):
        return {
            **action,
            "kind": "create",
            "entity": entity,
            "values": args,
            "summary": _new_summary(entity, args),
        }

    if name == "set_budget":
        return {**action, **_budget_action(args, messages)}

    row = _find_row(messages, entity, args.get("name"))
    if row is None:
        return {
            **action,
            "unresolved": True,
            "kind": kind,
            "entity": entity,
            "summary": f"{args.get('name')} (not in what you've read)",
        }

    action["if_modified_since"] = row.get("modified")
    action["entity"] = entity
    action["summary"] = _row_summary(entity, row)
    if kind == "delete":
        action["kind"] = "delete"
        action["values"] = _row_values(entity, row)
    else:
        action["kind"] = "update"
        action["changes"] = _diff(entity, row, args)
    return action


def _budget_action(args: dict, messages: list) -> dict:
    span = f"{args.get('category')}, {args.get('month')}/{args.get('year')}"
    current = _budget_amount(messages, args.get("category"))
    if current is not None:
        return {
            "kind": "update",
            "entity": "budget",
            "summary": f"Budget — {span}",
            "changes": [{"field": "amount", "from": current, "to": args.get("amount")}],
        }
    return {"kind": "create", "entity": "budget", "summary": f"Budget — {span}", "values": args}


def _diff(entity: str, row: dict, args: dict) -> list[dict]:
    label = _LABEL_FIELD[entity]
    current = {
        "amount": row.get("amount"),
        "date": _str_or_none(row.get("date")),
        "notes": row.get("notes"),
        label: row.get(f"{label}_name"),
    }
    changes = []
    for field, proposed in args.items():
        if field in ("name", "if_modified_since"):
            continue
        if field in current and current[field] != proposed:
            changes.append({"field": field, "from": current[field], "to": proposed})
    return changes


def _public(action: dict) -> dict:
    out = {k: action[k] for k in ("id", "tool", "kind", "entity", "summary")}
    if "changes" in action:
        out["changes"] = action["changes"]
    if "values" in action:
        out["values"] = action["values"]
    return out


# --- applying the approved subset ----------------------------------------


async def _apply(action: dict, approved: bool) -> ToolMessage:
    ref = {"id": action["id"], "name": action["tool"]}
    if not approved:
        return _tool_msg(ref, "Skipped by the member.")

    call_args = dict(action["call_args"])
    if action.get("if_modified_since"):
        call_args["if_modified_since"] = action["if_modified_since"]

    try:
        result = await tool_defs.BY_NAME[action["tool"]](**call_args)
    except FrappeError as exc:
        if _TIMESTAMP_MISMATCH in exc.detail:
            return _tool_msg(
                ref,
                f"{action['summary']} changed since you read it — re-read and re-propose "
                "if the change is still wanted.",
            )
        return _tool_msg(ref, f"Could not apply: {exc.detail}")

    return _tool_msg(ref, _applied_text(action, result))


def _applied_text(action: dict, result: Any) -> str:
    verb = {"create": "Created", "update": "Updated", "delete": "Deleted"}[action["kind"]]
    name = (result or {}).get("name") if isinstance(result, dict) else None
    return (
        f"{verb} {action['entity']} {name}".strip() if name else f"{verb} the {action['entity']}."
    )


# --- reading the tool history -------------------------------------------


def _find_row(messages: list, entity: str, name: str | None) -> dict | None:
    if not name:
        return None
    for rows in _tool_results(messages, _READ_FOR_ENTITY[entity]):
        for row in rows:
            if isinstance(row, dict) and row.get("name") == name:
                return row
    return None


def _budget_amount(messages: list, category: str | None) -> float | None:
    for rows in _tool_results(messages, "get_budgets"):
        for row in rows:
            if isinstance(row, dict) and row.get("category_name") == category:
                return row.get("budget_amount")
    return None


def _tool_results(messages: list, tool_name: str):
    """Every parsed result of `tool_name` in the thread, newest first."""
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == tool_name:
            parsed = _parse(message.content)
            if isinstance(parsed, list):
                yield parsed


def _parse(content: Any):
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return None
    return content


# --- small helpers ------------------------------------------------------


def _tool_msg(ref: dict, text: str) -> ToolMessage:
    return ToolMessage(content=text, tool_call_id=ref["id"], name=ref["name"])


def _new_summary(entity: str, args: dict) -> str:
    if entity in ("category", "source"):
        return f"Add {entity} {args.get('name')!r}"
    label = args.get(_LABEL_FIELD.get(entity, ""))
    parts = [str(args.get("amount"))]
    if label:
        parts.append(str(label))
    if args.get("date"):
        parts.append(str(args["date"]))
    return f"New {entity}: " + ", ".join(parts)


def _row_summary(entity: str, row: dict) -> str:
    label = row.get(f"{_LABEL_FIELD[entity]}_name") or "—"
    return f"{row.get('amount')} · {label} · {_str_or_none(row.get('date'))}"


def _row_values(entity: str, row: dict) -> dict:
    label = _LABEL_FIELD[entity]
    return {
        "amount": row.get("amount"),
        "date": _str_or_none(row.get("date")),
        label: row.get(f"{label}_name"),
        "notes": row.get("notes"),
    }


def _str_or_none(value: Any) -> str | None:
    return None if value is None else str(value)
