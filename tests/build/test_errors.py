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


def test_schema_not_found_messages() -> None:
    """SchemaNotFoundError exposes user-friendly message and hint."""
    error = SchemaNotFoundError("ast", "core.ast_nodes")

    assert "core.ast_nodes" in error.user_message
    assert "schema" in error.user_message
    assert "Add schema" in error.actionable_hint
    assert error.error_code == "SCHEMANOTFOUNDERROR"


def test_column_count_mismatch_hint() -> None:
    """ColumnCountMismatchError includes table, row, and expected counts."""
    error = ColumnCountMismatchError("metrics", "analytics.function_metrics", 5, 3, row_index=2)

    assert "row 2" in error.user_message
    assert "5 columns" in error.actionable_hint
    assert "metrics" in error.actionable_hint


def test_dependency_unavailable_hint() -> None:
    """DependencyUnavailableError returns descriptive hint."""
    error = DependencyUnavailableError("metrics", "ast", "failed validation")

    assert "metrics" in error.user_message
    assert "ast" in error.user_message
    assert "Fix the issue" in error.actionable_hint


def test_tool_not_available_hint_fallback() -> None:
    """ToolNotAvailableError returns install hint or fallback string."""
    error = ToolNotAvailableError("scip", "nonexistent-tool")

    assert "nonexistent-tool" in (error.actionable_hint or "")
    assert "not found" in error.user_message


def test_target_not_found_suggestions() -> None:
    """TargetNotFoundError suggests close matches when provided."""
    error = TargetNotFoundError("fuction_metrics", ["function_metrics", "ast"])

    assert "does not exist" in error.user_message
    hint = error.actionable_hint or ""
    assert "Did you mean" in hint
    assert "function_metrics" in hint


def test_missing_dependency_error_hint() -> None:
    """MissingDependencyError returns actionable guidance."""
    error = MissingDependencyError("metrics", "ast")

    assert "metrics" in error.user_message
    assert "ast" in error.actionable_hint


def test_target_timeout_hint() -> None:
    """TargetTimeoutError reports elapsed and limit values."""
    error = TargetTimeoutError("metrics", timeout_ms=500, elapsed_ms=750)

    assert "750ms" in error.user_message
    assert "500ms" in error.user_message
    assert "Increase execution.max_runtime_ms" in error.actionable_hint


def test_plugin_execution_error_chains_actionable_hint() -> None:
    """PluginExecutionError surfaces actionable hint from inner BuildError."""
    inner = ArtifactNotFoundError("metrics", "graph.json", path=Path("graph.json"))
    error = PluginExecutionError("metrics", "function_metrics", inner)

    assert "function_metrics" in str(error)
    assert error.actionable_hint == inner.actionable_hint
    assert "metrics" in error.user_message


def test_build_error_collection_operations() -> None:
    """BuildErrorCollection supports filtering, merging, and raising."""
    collection = BuildErrorCollection()
    first = SchemaNotFoundError("ast", "core.ast_nodes")
    second = ToolNotAvailableError("metrics", "pyright")
    collection.add(first)
    collection.add(second)
    collection.add_warning("Low disk space")

    assert bool(collection) is True
    assert len(collection) == 2
    assert collection.has_errors is True
    assert collection.has_warnings is True
    assert collection.by_type(ToolNotAvailableError) == [second]
    assert collection.by_target("ast") == [first]

    summary = collection.format_summary()
    assert "Build failed with 2 error" in summary
    assert "Warnings (1)" in summary
    assert "Hint:" in summary

    merged = collection.merge(BuildErrorCollection(errors=[first]))
    assert len(merged.errors) == 3

    with pytest.raises(SchemaNotFoundError):
        collection.raise_if_errors()


def test_error_collection_empty_summary() -> None:
    """Empty collections format to 'No errors' and do not raise."""
    collection = BuildErrorCollection()

    assert collection.format_summary() == "No errors"
    collection.raise_if_errors()  # Should not raise
