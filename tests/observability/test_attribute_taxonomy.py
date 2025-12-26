"""Attribute taxonomy guardrail tests."""

from __future__ import annotations

from codeintel.observability.attribute_sanitizer import limit_cli_arg_names, shape_attributes
from codeintel.observability.policy import ObservabilityPolicy


def test_filter_operation_attributes_drops_unknown_keys() -> None:
    """Unknown attribute keys should be filtered out."""
    attrs = {
        "http.method": "GET",
        "http.route": "/v1/semantic/query",
        "codeintel.output_format": "json",
        "unknown.key": "nope",
    }
    policy = ObservabilityPolicy()
    filtered = shape_attributes(attrs, allowed_keys=policy.operation_attribute_allowlist)
    assert "http.method" in filtered
    assert "http.route" in filtered
    assert "codeintel.output_format" in filtered
    assert "unknown.key" not in filtered


def test_filter_db_attributes_allows_db_and_codeintel_prefixes() -> None:
    """DB span attributes should allow db.* and codeintel.* prefixes."""
    attrs = {
        "db.system.name": "duckdb",
        "codeintel.repo": "demo/repo",
        "cli.command": "build",
    }
    policy = ObservabilityPolicy()
    filtered = shape_attributes(attrs, allowed_prefixes=policy.db_attribute_prefixes)
    assert "db.system.name" in filtered
    assert "codeintel.repo" in filtered
    assert "cli.command" not in filtered


def test_limit_cli_arg_names_truncates_long_lists() -> None:
    """CLI arg name lists should be truncated to the budget."""
    policy = ObservabilityPolicy()
    max_len = policy.budget.cli_arg_names_max
    names = tuple(f"arg{i}" for i in range(max_len + 3))
    bounded = limit_cli_arg_names(names, max_len=max_len)
    assert len(bounded) == max_len
    assert bounded[0] == "arg0"
