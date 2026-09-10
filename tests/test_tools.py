import httpx
import pytest
import respx

from expenso_assistant import tools

from .conftest import API, method_url

pytestmark = pytest.mark.usefixtures("bound_client")


@respx.mock
async def test_read_calls_frappe_as_the_member_with_no_family_filter():
    route = respx.get(method_url(f"{API}.get_expenses")).mock(
        return_value=httpx.Response(200, json={"message": [{"name": "e1", "amount": 5}]})
    )

    result = await tools.get_expenses(month=6, year=2025)

    assert result == [{"name": "e1", "amount": 5}]
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer member-bearer"
    # The Member's bearer is the only scoping input — the tool adds no `family`.
    assert dict(request.url.params) == {"month": "6", "year": "2025"}


@respx.mock
async def test_read_with_crafted_params_still_carries_only_month_year():
    route = respx.get(method_url(f"{API}.get_analytics")).mock(
        return_value=httpx.Response(200, json={"message": {"total": 0}})
    )

    await tools.get_analytics(month=1, year=2030)

    assert set(dict(route.calls.last.request.url.params)) == {"month", "year"}


@respx.mock
async def test_create_expense_posts_entry_method_from_context():
    route = respx.post(method_url(f"{API}.create_expense")).mock(
        return_value=httpx.Response(200, json={"message": {"name": "e9"}})
    )

    token = tools.bind_entry_method("connector")
    try:
        await tools.create_expense(amount=12.5, notes="Lunch")
    finally:
        tools.reset_entry_method(token)

    body = respx.calls.last.request.read()
    import json

    payload = json.loads(body)
    assert payload["entry_method"] == "connector"
    assert payload["amount"] == 12.5
    assert payload["notes"] == "Lunch"
    # None-valued optionals are dropped, not sent as null.
    assert "date" not in payload and "category" not in payload
    assert route.called


@respx.mock
async def test_write_raises_on_frappe_error():
    respx.post(method_url(f"{API}.create_income")).mock(
        return_value=httpx.Response(409, text="conflict")
    )

    with pytest.raises(Exception) as excinfo:
        await tools.create_income(amount=1)
    assert "409" in str(excinfo.value)


def test_tool_sets_are_partitioned_read_vs_write():
    read_names = {fn.__name__ for fn in tools.READ_TOOLS}
    write_names = {fn.__name__ for fn in tools.WRITE_TOOLS}

    assert read_names == {
        "get_expenses",
        "get_income",
        "get_analytics",
        "get_budgets",
        "list_categories",
        "list_sources",
    }
    assert write_names == {
        "create_expense",
        "update_expense",
        "delete_expense",
        "create_income",
        "update_income",
        "delete_income",
        "add_category",
        "add_source",
        "set_budget",
    }
    assert read_names.isdisjoint(write_names)
    # The agent must never be able to rename/delete a Category or Source.
    assert not (read_names | write_names) & {
        "rename_category",
        "delete_category",
        "rename_source",
        "delete_source",
    }
