from expenso_assistant.config import MODEL_PRICING, cost_for, get_settings


def test_default_model_is_priced():
    assert get_settings().openai_model in MODEL_PRICING


def test_cost_for_applies_the_rate_table_per_token_class():
    # 1M uncached input + 1M output on gpt-4o-mini = 0.15 + 0.60
    cost = cost_for("gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
    assert round(cost, 6) == 0.75


def test_cost_for_discounts_cached_input_and_bills_reasoning_as_output():
    cost = cost_for(
        "gpt-4o-mini",
        input_tokens=1_000_000,
        cached_tokens=1_000_000,
        output_tokens=0,
        reasoning_tokens=1_000_000,
    )
    # all input cached (0.075) + reasoning billed as output (0.60)
    assert round(cost, 6) == 0.675


def test_cost_for_unknown_model_is_none():
    assert cost_for("some-unlisted-model", input_tokens=10, output_tokens=10) is None
