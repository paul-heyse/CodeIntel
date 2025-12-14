"""PR51: Tests for validation reporters native Hamilton migration.

This module tests the migration of validation reporter flush() methods to
Hamilton native materialization helpers. It verifies:
1. to_rows() methods return correct tuple format
2. flush() methods emit DeprecationWarning
3. Materialization helpers integrate with Hamilton build layer
4. Both tables are populated with correct schemas
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.parsing import (
    FunctionValidationReporter,
    GraphValidationReporter,
    ValidationResult,
    get_validation_rows,
    materialize_function_validation,
    materialize_graph_validation,
)
from codeintel.analytics.parsing.validation import (
    FUNCTION_VALIDATION_COLS,
    GRAPH_VALIDATION_COLS,
)
from codeintel.build.hamilton.native.materializer import (
    MaterializationContext,
    materialize_rows,
)
from codeintel.config.primitives import SnapshotRef
from tests.build.hamilton.test_pr50_architecture_guardrails import ALLOWLIST_IBIS_WRITE_FILES

if TYPE_CHECKING:
    from tests._helpers import TestContext


EXPECTED_FUNCTION_VALIDATION_COLS = 8
EXPECTED_GRAPH_VALIDATION_COLS = 10
EXPECTED_ROW_COUNT_SINGLE = 1
EXPECTED_ROW_COUNT_EMPTY = 0


# =============================================================================
# Tests for to_rows() methods
# =============================================================================


def test_function_reporter_to_rows_returns_tuples() -> None:
    """Verify FunctionValidationReporter.to_rows() returns tuple of tuples."""
    reporter = FunctionValidationReporter(repo="org/repo", commit="abc123")
    reporter.record(
        function_goid_h128=12345,
        rel_path="pkg/module.py",
        qualname="my_function",
        issue="parse_failed",
        detail="Syntax error at line 10",
    )

    rows = reporter.to_rows()

    if not isinstance(rows, tuple):
        pytest.fail(f"Expected tuple, got {type(rows)}")
    if len(rows) != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected {EXPECTED_ROW_COUNT_SINGLE} row, got {len(rows)}")
    if not isinstance(rows[0], tuple):
        pytest.fail(f"Expected row to be tuple, got {type(rows[0])}")


def test_graph_reporter_to_rows_returns_tuples() -> None:
    """Verify GraphValidationReporter.to_rows() returns tuple of tuples."""
    reporter = GraphValidationReporter(repo="org/repo", commit="abc123")
    reporter.record(
        graph_name="call_graph",
        issue="orphan_edge",
        detail="Edge references missing node",
        entity_id="edge_001",
        extras={"severity": "warning", "rel_path": "pkg/module.py"},
    )

    rows = reporter.to_rows()

    if not isinstance(rows, tuple):
        pytest.fail(f"Expected tuple, got {type(rows)}")
    if len(rows) != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected {EXPECTED_ROW_COUNT_SINGLE} row, got {len(rows)}")
    if not isinstance(rows[0], tuple):
        pytest.fail(f"Expected row to be tuple, got {type(rows[0])}")


def test_function_reporter_to_rows_empty() -> None:
    """Verify FunctionValidationReporter.to_rows() returns empty tuple when no rows."""
    reporter = FunctionValidationReporter(repo="org/repo", commit="abc123")
    rows = reporter.to_rows()

    if len(rows) != EXPECTED_ROW_COUNT_EMPTY:
        pytest.fail(f"Expected empty tuple, got {len(rows)} rows")


def test_graph_reporter_to_rows_empty() -> None:
    """Verify GraphValidationReporter.to_rows() returns empty tuple when no rows."""
    reporter = GraphValidationReporter(repo="org/repo", commit="abc123")
    rows = reporter.to_rows()

    if len(rows) != EXPECTED_ROW_COUNT_EMPTY:
        pytest.fail(f"Expected empty tuple, got {len(rows)} rows")


# =============================================================================
# Tests for get_validation_rows helper
# =============================================================================


def test_get_validation_rows_returns_result() -> None:
    """Verify get_validation_rows returns ValidationResult."""
    func_reporter = FunctionValidationReporter(repo="org/repo", commit="abc123")
    graph_reporter = GraphValidationReporter(repo="org/repo", commit="abc123")

    result = get_validation_rows(func_reporter, graph_reporter)

    if not isinstance(result, ValidationResult):
        pytest.fail(f"Expected ValidationResult, got {type(result)}")


def test_get_validation_rows_handles_none() -> None:
    """Verify get_validation_rows handles None reporters."""
    result = get_validation_rows(None, None)

    if result.function_rows != ():
        pytest.fail("Expected empty function_rows for None reporter")
    if result.graph_rows != ():
        pytest.fail("Expected empty graph_rows for None reporter")


# =============================================================================
# Tests for flush() deprecation warnings
# =============================================================================


def test_function_reporter_flush_deprecation(test_ctx: TestContext) -> None:
    """Verify FunctionValidationReporter.flush() emits DeprecationWarning."""
    reporter = FunctionValidationReporter(repo=test_ctx.repo, commit=test_ctx.commit)

    with pytest.warns(DeprecationWarning, match="FunctionValidationReporter.flush is deprecated"):
        reporter.flush(test_ctx.gateway)


def test_graph_reporter_flush_deprecation(test_ctx: TestContext) -> None:
    """Verify GraphValidationReporter.flush() emits DeprecationWarning."""
    reporter = GraphValidationReporter(repo=test_ctx.repo, commit=test_ctx.commit)

    with pytest.warns(DeprecationWarning, match="GraphValidationReporter.flush is deprecated"):
        reporter.flush(test_ctx.gateway)


# =============================================================================
# Tests for materialization helpers
# =============================================================================


def test_materialize_function_validation_writes(test_ctx: TestContext) -> None:
    """Verify materialize_function_validation writes rows to database."""
    test_ctx.gateway.policy.ensure_table("analytics.function_validation")

    reporter = FunctionValidationReporter(repo=test_ctx.repo, commit=test_ctx.commit)
    reporter.record(
        function_goid_h128=12345,
        rel_path="pkg/module.py",
        qualname="my_function",
        issue="parse_failed",
        detail="Syntax error at line 10",
    )

    ctx = MaterializationContext(
        gateway=test_ctx.gateway,
        snapshot=SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(),
        ),
        validate=False,
    )

    ref = materialize_function_validation(ctx, reporter)

    if ref.row_count != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_SINGLE}, got {ref.row_count}")

    count = test_ctx.con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.function_validation
        WHERE repo = ? AND commit = ?
        """,
        [test_ctx.repo, test_ctx.commit],
    ).fetchone()
    if count is None or count[0] != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected {EXPECTED_ROW_COUNT_SINGLE} row in DB, got {count}")


def test_materialize_graph_validation_writes(test_ctx: TestContext) -> None:
    """Verify materialize_graph_validation writes rows to database."""
    test_ctx.gateway.policy.ensure_table("analytics.graph_validation")

    reporter = GraphValidationReporter(repo=test_ctx.repo, commit=test_ctx.commit)
    reporter.record(
        graph_name="call_graph",
        issue="orphan_edge",
        detail="Edge references missing node",
        entity_id="edge_001",
        extras={"severity": "warning", "rel_path": "pkg/module.py"},
    )

    ctx = MaterializationContext(
        gateway=test_ctx.gateway,
        snapshot=SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(),
        ),
        validate=False,
    )

    ref = materialize_graph_validation(ctx, reporter)

    if ref.row_count != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_SINGLE}, got {ref.row_count}")

    count = test_ctx.con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.graph_validation
        WHERE repo = ? AND commit = ?
        """,
        [test_ctx.repo, test_ctx.commit],
    ).fetchone()
    if count is None or count[0] != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected {EXPECTED_ROW_COUNT_SINGLE} row in DB, got {count}")


def test_materialize_rows_handles_empty_function_validation(test_ctx: TestContext) -> None:
    """Verify materialize_rows handles empty function validation rows."""
    test_ctx.gateway.policy.ensure_table("analytics.function_validation")

    ctx = MaterializationContext(
        gateway=test_ctx.gateway,
        snapshot=SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(),
        ),
        validate=False,
    )

    ref = materialize_rows(
        ctx,
        "analytics.function_validation",
        [],
        FUNCTION_VALIDATION_COLS,
    )

    if ref.row_count != EXPECTED_ROW_COUNT_EMPTY:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_EMPTY}, got {ref.row_count}")


# =============================================================================
# Architecture guardrail tests
# =============================================================================


def test_validation_in_allowlist() -> None:
    """Verify analytics/parsing/validation.py is in allowlist for backward compat.

    The deprecated flush() methods still have direct DB writes
    for backward compatibility. Once the methods are removed, the file
    should be removed from the allowlist.

    New code should use to_rows() + materialize_*() helpers instead.
    """
    if "src/codeintel/analytics/parsing/validation.py" not in ALLOWLIST_IBIS_WRITE_FILES:
        pytest.fail(
            "analytics/parsing/validation.py should remain in "
            "ALLOWLIST_IBIS_WRITE_FILES until deprecated flush() methods are removed"
        )


# =============================================================================
# Column count tests
# =============================================================================


def test_function_validation_cols_count() -> None:
    """Verify FUNCTION_VALIDATION_COLS has expected column count."""
    actual_count = len(FUNCTION_VALIDATION_COLS)
    if actual_count != EXPECTED_FUNCTION_VALIDATION_COLS:
        pytest.fail(
            f"Expected {EXPECTED_FUNCTION_VALIDATION_COLS} columns in "
            f"FUNCTION_VALIDATION_COLS, got {actual_count}"
        )


def test_graph_validation_cols_count() -> None:
    """Verify GRAPH_VALIDATION_COLS has expected column count."""
    actual_count = len(GRAPH_VALIDATION_COLS)
    if actual_count != EXPECTED_GRAPH_VALIDATION_COLS:
        pytest.fail(
            f"Expected {EXPECTED_GRAPH_VALIDATION_COLS} columns in "
            f"GRAPH_VALIDATION_COLS, got {actual_count}"
        )
