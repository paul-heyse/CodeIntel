"""Attribute taxonomy guardrail tests."""

from __future__ import annotations

from codeintel.observability.attribute_sanitizer import limit_cli_arg_names
from codeintel.observability.attribute_schema import build_attribute_normalizer
from codeintel.observability.policy import ObservabilityPolicy
from codeintel.observability.semconv_keys import (
    CODEINTEL_OUTPUT_FORMAT,
    CODEINTEL_REPO,
    DB_SYSTEM_NAME,
    HTTP_METHOD,
    HTTP_ROUTE,
)


def test_filter_operation_attributes_drops_unknown_keys() -> None:
    """Unknown attribute keys should be filtered out."""
    attrs = {
        HTTP_METHOD: "GET",
        HTTP_ROUTE: "/v1/semantic/query",
        CODEINTEL_OUTPUT_FORMAT: "json",
        "unknown.key": "nope",
    }
    policy = ObservabilityPolicy()
    normalizer = build_attribute_normalizer(policy)
    filtered = normalizer.normalize(attrs, allowed_keys=policy.operation_attribute_allowlist)
    assert HTTP_METHOD in filtered
    assert HTTP_ROUTE in filtered
    assert CODEINTEL_OUTPUT_FORMAT in filtered
    assert "unknown.key" not in filtered


def test_filter_db_attributes_allows_db_and_codeintel_prefixes() -> None:
    """DB span attributes should allow db.* and codeintel.* prefixes."""
    attrs = {
        DB_SYSTEM_NAME: "duckdb",
        CODEINTEL_REPO: "demo/repo",
        "cli.command": "build",
    }
    policy = ObservabilityPolicy()
    normalizer = build_attribute_normalizer(policy)
    filtered = normalizer.normalize(attrs, allowed_prefixes=policy.db_attribute_prefixes)
    assert DB_SYSTEM_NAME in filtered
    assert CODEINTEL_REPO in filtered
    assert "cli.command" not in filtered


def test_limit_cli_arg_names_truncates_long_lists() -> None:
    """CLI arg name lists should be truncated to the budget."""
    policy = ObservabilityPolicy()
    max_len = policy.budget.cli_arg_names_max
    names = tuple(f"arg{i}" for i in range(max_len + 3))
    bounded = limit_cli_arg_names(names, max_len=max_len)
    assert len(bounded) == max_len
    assert bounded[0] == "arg0"
