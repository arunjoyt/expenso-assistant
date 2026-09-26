"""The confirm card's text for each proposed write (ADR 0010).

A `description` callback for the stock `HumanInTheLoopMiddleware`: one line
per write call, read from the rows already in the tool history (no re-read).
Updates show `field: from → to`, so the member sees what changes, not only
the new values.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage

from .. import tools as tool_defs

_READ_FOR_ENTITY = {"expense": "get_expenses", "income": "get_income"}
_LABEL_FIELD = {"expense": "category", "income": "source"}
_CREATE_TOOLS = ("create_expense", "create_income", "add_category", "add_source")


def describe_write(tool_call: dict, state: dict, runtime: Any) -> str:
    name = tool_call["name"]
    args = {k: v for k, v in tool_call["args"].items() if v is not None}
    _, entity = tool_defs.write_kind(name)
    messages = state.get("messages") or []

    if name in _CREATE_TOOLS:
        return _new_summary(entity, args)
    if name == "set_budget":
        return _budget_summary(args, messages)

    row = _find_row(messages, entity, args.get("name"))
    if name.startswith("delete_"):
        return f"Delete {entity} {_row_label(entity, row, args)}"
    changes = _diff(entity, row, args) if row else _proposed(args)
    return f"Edit {entity} {_row_label(entity, row, args)} — {changes or 'no field changes'}"


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


def _budget_summary(args: dict, messages: list) -> str:
    span = f"{args.get('category')}, {args.get('month')}/{args.get('year')}"
    current = _budget_amount(messages, args.get("category"))
    if current is None:
        return f"New budget — {span}: {args.get('amount')}"
    return f"Budget — {span}: amount {current} → {args.get('amount')}"


def _row_label(entity: str, row: dict | None, args: dict) -> str:
    if row is None:
        return f"{args.get('name')} (not in what you've read)"
    label = row.get(f"{_LABEL_FIELD[entity]}_name") or "—"
    return f"{row.get('amount')} · {label} · {_str_or_none(row.get('date'))}"


def _diff(entity: str, row: dict, args: dict) -> str:
    label = _LABEL_FIELD[entity]
    current = {
        "amount": row.get("amount"),
        "date": _str_or_none(row.get("date")),
        "notes": row.get("notes"),
        label: row.get(f"{label}_name"),
    }
    return "; ".join(
        f"{field}: {current[field]} → {proposed}"
        for field, proposed in args.items()
        if field in current and current[field] != proposed
    )


def _proposed(args: dict) -> str:
    return "; ".join(
        f"{field}: → {value}"
        for field, value in args.items()
        if field not in ("name", "if_modified_since")
    )


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


def _str_or_none(value: Any) -> str | None:
    return None if value is None else str(value)
