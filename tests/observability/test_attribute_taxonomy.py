"""Attribute taxonomy guardrail tests."""

from __future__ import annotations

from codeintel.observability.attribute_taxonomy import (
    CLI_ARG_NAMES_MAX,
    filter_db_attributes,
    filter_operation_attributes,
    limit_cli_arg_names,
)


def test_filter_operation_attributes_drops_unknown_keys() -> None:
    """Unknown attribute keys should be filtered out."""
    attrs = {
        "http.method": "GET",
        "http.route": "/v1/semantic/query",
        "codeintel.output_format": "json",
        "unknown.key": "nope",
    }
    filtered = filter_operation_attributes(attrs)
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
    filtered = filter_db_attributes(attrs)
    assert "db.system.name" in filtered
    assert "codeintel.repo" in filtered
    assert "cli.command" not in filtered


def test_limit_cli_arg_names_truncates_long_lists() -> None:
    """CLI arg name lists should be truncated to the budget."""
    names = tuple(f"arg{i}" for i in range(CLI_ARG_NAMES_MAX + 3))
    bounded = limit_cli_arg_names(names)
    assert len(bounded) == CLI_ARG_NAMES_MAX
    assert bounded[0] == "arg0"
