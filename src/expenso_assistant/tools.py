"""The one tool definition — typed async functions over Frappe REST.

Two consumers bind to these, and only these:

- the **LangGraph agent** (P6-S5) binds the functions directly as LangChain
  tools — no MCP in the agent's path;
- the **FastMCP server** (`mcp_server.py`) registers the same functions for
  external ChatGPT/Claude connectors, with elicitation on the writes.

Both derive their schema from these signatures. Family-scoping is Frappe's job
(bearer passthrough + permission hooks); these functions build no filters of
their own and cannot construct a raw query (ADR 0008).

`rename_category` / `delete_category` / `rename_source` / `delete_source` are
deliberately absent — the agent may *recommend* them but never execute them
(they rewrite the meaning of historical rows or cascade-delete Budgets; their
safety comes from being done on Settings with the whole list in view).
"""

from __future__ import annotations

import contextvars

from .frappe_client import current_frappe_client

_API = "expenso.expenso.api"

# Provenance stamped on every write, taken from the caller's context: the
# FastMCP connector adapter binds "connector"; the agent binds "assistant" or
# "receipt" per feature (P6-S5/S7/P7-S1). Frappe coerces anything unexpected to
# "manual".
_entry_method: contextvars.ContextVar[str] = contextvars.ContextVar(
    "entry_method", default="assistant"
)


def bind_entry_method(value: str) -> contextvars.Token:
    return _entry_method.set(value)


def reset_entry_method(token: contextvars.Token) -> None:
    _entry_method.reset(token)


# --- reads -------------------------------------------------------------------


async def get_expenses(month: int, year: int) -> list[dict]:
    """List the Family's Expenses for a calendar month (month 1-12, four-digit year)."""
    return await current_frappe_client().call(f"{_API}.get_expenses", month=month, year=year)


async def get_income(month: int, year: int) -> list[dict]:
    """List the Family's Income entries for a calendar month."""
    return await current_frappe_client().call(f"{_API}.get_income", month=month, year=year)


async def get_analytics(month: int, year: int) -> dict:
    """The Family's monthly totals, per-Category breakdown and Budget status."""
    return await current_frappe_client().call(f"{_API}.get_analytics", month=month, year=year)


async def get_budgets(month: int, year: int) -> list[dict]:
    """The Family's Budget amount per Category for a month."""
    return await current_frappe_client().call(f"{_API}.get_budgets", month=month, year=year)


async def list_categories() -> list[str]:
    """The Family's Category names — validate a `category` against these before a write."""
    return await current_frappe_client().call(f"{_API}.list_categories")


async def list_sources() -> list[str]:
    """The Family's Source names — validate a `source` against these before a write."""
    return await current_frappe_client().call(f"{_API}.list_sources")


# --- writes ----------------------------------------------------------------


async def create_expense(
    amount: float,
    date: str | None = None,
    category: str | None = None,
    notes: str | None = None,
    message: str | None = None,
) -> dict:
    """Create an Expense. `category` is matched to an existing Category by name
    or label; an unmatched value is left unset. `message` is stored for audit
    only on external-connector writes (never shown in the app) — omit it for
    in-app Assistant writes, which the Member has already confirmed."""
    return await current_frappe_client().call(
        f"{_API}.create_expense",
        write=True,
        amount=amount,
        date=date,
        category=category,
        notes=notes,
        entry_method=_entry_method.get(),
        external_message=message,
    )


async def update_expense(
    name: str,
    amount: float | None = None,
    date: str | None = None,
    category: str | None = None,
    notes: str | None = None,
    if_modified_since: str | None = None,
) -> dict:
    """Edit an Expense. Pass `if_modified_since` (the row's `modified` value from
    your last read) so a row changed underneath you is rejected, not clobbered."""
    return await current_frappe_client().call(
        f"{_API}.update_expense",
        write=True,
        name=name,
        amount=amount,
        date=date,
        category=category,
        notes=notes,
        if_modified_since=if_modified_since,
    )


async def delete_expense(name: str, if_modified_since: str | None = None) -> None:
    """Delete an Expense. Pass `if_modified_since` as for `update_expense`."""
    return await current_frappe_client().call(
        f"{_API}.delete_expense", write=True, name=name, if_modified_since=if_modified_since
    )


async def create_income(
    amount: float,
    date: str | None = None,
    source: str | None = None,
    notes: str | None = None,
    message: str | None = None,
) -> dict:
    """Create an Income. `source` is matched to an existing Source by name or
    label; an unmatched value is left unset. `message` is audit-only on
    external-connector writes (see `create_expense`)."""
    return await current_frappe_client().call(
        f"{_API}.create_income",
        write=True,
        amount=amount,
        date=date,
        source=source,
        notes=notes,
        entry_method=_entry_method.get(),
        external_message=message,
    )


async def update_income(
    name: str,
    amount: float | None = None,
    date: str | None = None,
    source: str | None = None,
    notes: str | None = None,
    if_modified_since: str | None = None,
) -> dict:
    """Edit an Income. Pass `if_modified_since` as for `update_expense`."""
    return await current_frappe_client().call(
        f"{_API}.update_income",
        write=True,
        name=name,
        amount=amount,
        date=date,
        source=source,
        notes=notes,
        if_modified_since=if_modified_since,
    )


async def delete_income(name: str, if_modified_since: str | None = None) -> None:
    """Delete an Income. Pass `if_modified_since` as for `update_expense`."""
    return await current_frappe_client().call(
        f"{_API}.delete_income", write=True, name=name, if_modified_since=if_modified_since
    )


async def add_category(name: str) -> dict:
    """Create a new Category in the Family."""
    return await current_frappe_client().call(f"{_API}.add_category", write=True, name=name)


async def add_source(name: str) -> dict:
    """Create a new Source in the Family."""
    return await current_frappe_client().call(f"{_API}.add_source", write=True, name=name)


async def set_budget(
    category: str, month: int, year: int, amount: float | None = None
) -> dict | None:
    """Set (or, with amount omitted, clear) a Category's Budget for a month."""
    return await current_frappe_client().call(
        f"{_API}.set_budget",
        write=True,
        category=category,
        month=month,
        year=year,
        amount=amount,
    )


READ_TOOLS = (
    get_expenses,
    get_income,
    get_analytics,
    get_budgets,
    list_categories,
    list_sources,
)

WRITE_TOOLS = (
    create_expense,
    update_expense,
    delete_expense,
    create_income,
    update_income,
    delete_income,
    add_category,
    add_source,
    set_budget,
)

ALL_TOOLS = READ_TOOLS + WRITE_TOOLS
WRITE_TOOL_NAMES = frozenset(fn.__name__ for fn in WRITE_TOOLS)
BY_NAME = {fn.__name__: fn for fn in ALL_TOOLS}

# How a proposed write reads on the confirm card, and whether a stale-write
# guard applies (creates have no prior row).
_WRITE_KIND = {
    "create_expense": ("create", "expense"),
    "update_expense": ("update", "expense"),
    "delete_expense": ("delete", "expense"),
    "create_income": ("create", "income"),
    "update_income": ("update", "income"),
    "delete_income": ("delete", "income"),
    "add_category": ("create", "category"),
    "add_source": ("create", "source"),
    "set_budget": ("update", "budget"),  # flips to "create" when no row exists yet
}


def write_kind(tool_name: str) -> tuple[str, str]:
    """(kind, entity) for a write tool — `kind` in create / update / delete."""
    return _WRITE_KIND[tool_name]
