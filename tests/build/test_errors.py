"""Unit tests for build error types and collections."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.build.errors import (
    ArtifactNotFoundError,
    BuildErrorCollection,
    ColumnCountMismatchError,
    DependencyUnavailableError,
    MissingDependencyError,
    PluginExecutionError,
    SchemaNotFoundError,
    TargetNotFoundError,
    TargetTimeoutError,
    ToolNotAvailableError,
)
from tests._helpers.assertions import expect_equal, expect_in, expect_true


def test_schema_not_found_messages() -> None:
    """SchemaNotFoundError exposes user-friendly message and hint."""
    error = SchemaNotFoundError("ast", "core.ast_nodes")

    expect_in("core.ast_nodes", error.user_message)
    expect_in("schema", error.user_message)
    expect_in("Add schema", error.actionable_hint)
    expect_equal(error.error_code, "SCHEMANOTFOUNDERROR")


def test_column_count_mismatch_hint() -> None:
    """ColumnCountMismatchError includes table, row, and expected counts."""
    error = ColumnCountMismatchError("metrics", "analytics.function_metrics", 5, 3, row_index=2)

    expect_in("row 2", error.user_message)
    expect_in("5 columns", error.actionable_hint)
    expect_in("metrics", error.actionable_hint)


def test_dependency_unavailable_hint() -> None:
    """DependencyUnavailableError returns descriptive hint."""
    error = DependencyUnavailableError("metrics", "ast", "failed validation")

    expect_in("metrics", error.user_message)
    expect_in("ast", error.user_message)
    expect_in("Fix the issue", error.actionable_hint)


def test_tool_not_available_hint_fallback() -> None:
    """ToolNotAvailableError returns install hint or fallback string."""
    error = ToolNotAvailableError("scip", "nonexistent-tool")

    expect_in("nonexistent-tool", error.actionable_hint or "")
    expect_in("not found", error.user_message)


def test_target_not_found_suggestions() -> None:
    """TargetNotFoundError suggests close matches when provided."""
    error = TargetNotFoundError("fuction_metrics", ["function_metrics", "ast"])

    expect_in("does not exist", error.user_message)
    hint = error.actionable_hint or ""
    expect_in("Did you mean", hint)
    expect_in("function_metrics", hint)


def test_missing_dependency_error_hint() -> None:
    """MissingDependencyError returns actionable guidance."""
    error = MissingDependencyError("metrics", "ast")

    expect_in("metrics", error.user_message)
    expect_in("ast", error.actionable_hint)


def test_target_timeout_hint() -> None:
    """TargetTimeoutError reports elapsed and limit values."""
    error = TargetTimeoutError("metrics", timeout_ms=500, elapsed_ms=750)

    expect_in("750ms", error.user_message)
    expect_in("500ms", error.user_message)
    expect_in("Increase execution.max_runtime_ms", error.actionable_hint)


def test_plugin_execution_error_chains_actionable_hint() -> None:
    """PluginExecutionError surfaces actionable hint from inner BuildError."""
    inner = ArtifactNotFoundError("metrics", "graph.json", path=Path("graph.json"))
    error = PluginExecutionError("metrics", "function_metrics", inner)

    expect_in("function_metrics", str(error))
    expect_equal(error.actionable_hint, inner.actionable_hint)
    expect_in("metrics", error.user_message)


def test_build_error_collection_operations() -> None:
    """BuildErrorCollection supports filtering, merging, and raising."""
    collection = BuildErrorCollection()
    first = SchemaNotFoundError("ast", "core.ast_nodes")
    second = ToolNotAvailableError("metrics", "pyright")
    collection.add(first)
    collection.add(second)
    collection.add_warning("Low disk space")

    expect_true(bool(collection))
    expect_equal(len(collection), 2)
    expect_true(collection.has_errors)
    expect_true(collection.has_warnings)
    expect_equal(collection.by_type(ToolNotAvailableError), [second])
    expect_equal(collection.by_target("ast"), [first])

    summary = collection.format_summary()
    expect_in("Build failed with 2 error", summary)
    expect_in("Warnings (1)", summary)
    expect_in("Hint:", summary)

    merged = collection.merge(BuildErrorCollection(errors=[first]))
    expect_equal(len(merged.errors), 3)

    with pytest.raises(SchemaNotFoundError):
        collection.raise_if_errors()


def test_error_collection_empty_summary() -> None:
    """Empty collections format to 'No errors' and do not raise."""
    collection = BuildErrorCollection()

    expect_equal(collection.format_summary(), "No errors")
    collection.raise_if_errors()  # Should not raise
