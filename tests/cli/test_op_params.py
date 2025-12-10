"""Tests for op_params coercion utilities."""

from __future__ import annotations

from codeintel.cli.introspection import (
    CliParamSpec,
    coerce_params_from_strings,
    coerce_string_param,
)
from tests._helpers.assertions import expect_equal, expect_true


def test_coerce_string_param_basic_types() -> None:
    """String tunnel coercion handles primitives."""
    expect_equal(coerce_string_param("42", int), 42)
    expect_equal(coerce_string_param("3.14", float), 3.14)
    expect_true(coerce_string_param("true", bool))
    expect_equal(coerce_string_param("value", None), "value")


def test_coerce_params_from_strings_skips_none() -> None:
    """None values are filtered out and types coerced."""
    specs = (
        CliParamSpec(
            name="limit",
            cli_name="limit",
            python_type=int,
            default=None,
            role="filter",
            help_text="limit",
            help_panel="Filters",
            is_optional=True,
        ),
        CliParamSpec(
            name="q",
            cli_name="q",
            python_type=str,
            default=None,
            role="selector",
            help_text="q",
            help_panel="Selectors",
            is_optional=True,
        ),
    )
    raw = {"limit": "5", "q": None}
    result = coerce_params_from_strings(raw, specs)
    expect_equal(result, {"limit": 5})
