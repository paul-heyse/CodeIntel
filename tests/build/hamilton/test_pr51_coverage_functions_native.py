"""PR51: Tests for coverage_functions native Hamilton migration.

This module tests the migration of compute_coverage_functions to
Hamilton native materialization. It verifies:
1. build_coverage_functions_expr returns correct Ibis expression
2. compute_coverage_functions emits DeprecationWarning
3. materialize_table writes coverage_functions to database
4. Native module exports are correct
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.compute.coverage import (
    build_coverage_functions_expr,
    compute_coverage_functions,
)
from codeintel.build.hamilton.native.materializer import (
    MaterializationContext,
    materialize_table,
)
from codeintel.config.primitives import SnapshotRef
from tests.build.hamilton.test_pr50_architecture_guardrails import ALLOWLIST_IBIS_WRITE_FILES

if TYPE_CHECKING:
    from tests._helpers import TestContext


EXPECTED_COVERAGE_FUNCTIONS_COLS = 16
EXPECTED_ROW_COUNT_SINGLE = 1
EXPECTED_ROW_COUNT_EMPTY = 0
EXPECTED_EXECUTABLE_LINES = 2
EXPECTED_COVERED_LINES = 1


# =============================================================================
# Tests for build_coverage_functions_expr
# =============================================================================


def test_build_coverage_functions_expr_returns_table_with_empty_data(
    test_ctx: TestContext,
) -> None:
    """Verify build_coverage_functions_expr returns table when data is empty."""
    # Tables exist (created by apply_all_schemas) but have no data for snapshot
    result = build_coverage_functions_expr(test_ctx.gateway, test_ctx.snapshot)

    # Should return an Ibis expression (empty but valid)
    if result is None:
        pytest.fail("Expected Ibis table expression, got None")


def test_build_coverage_functions_expr_returns_ibis_table(test_ctx: TestContext) -> None:
    """Verify build_coverage_functions_expr returns Ibis table when tables exist."""
    # Tables already exist (created by apply_all_schemas in test context)
    result = build_coverage_functions_expr(test_ctx.gateway, test_ctx.snapshot)

    if result is None:
        pytest.fail("Expected Ibis table expression, got None")


def test_build_coverage_functions_expr_with_data(test_ctx: TestContext) -> None:
    """Verify build_coverage_functions_expr produces correct rows with data."""
    # Tables already exist (created by apply_all_schemas in test context)
    # Insert test data using parameterized queries with correct schema
    # core.goids: goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line, created_at
    test_ctx.con.execute(
        """
        INSERT INTO core.goids (goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line, created_at)
        VALUES (12345, 'urn:func', ?, ?, 'foo.py', 'python', 'function', 'foo', 1, 3, NOW())
        """,
        [test_ctx.repo, test_ctx.commit],
    )
    # analytics.coverage_lines: repo, commit, rel_path, line, is_executable, is_covered, hits, context_count, created_at
    test_ctx.con.execute(
        """
        INSERT INTO analytics.coverage_lines (repo, commit, rel_path, line, is_executable, is_covered, hits, context_count, created_at)
        VALUES
            (?, ?, 'foo.py', 1, TRUE, TRUE, 1, 0, NOW()),
            (?, ?, 'foo.py', 2, TRUE, FALSE, 0, 0, NOW()),
            (?, ?, 'foo.py', 3, FALSE, FALSE, 0, 0, NOW())
        """,
        [
            test_ctx.repo,
            test_ctx.commit,
            test_ctx.repo,
            test_ctx.commit,
            test_ctx.repo,
            test_ctx.commit,
        ],
    )

    result = build_coverage_functions_expr(test_ctx.gateway, test_ctx.snapshot)

    if result is None:
        pytest.fail("Expected Ibis table expression, got None")

    # Execute the expression and check results
    df = result.execute()
    if len(df) != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected {EXPECTED_ROW_COUNT_SINGLE} row, got {len(df)}")

    row = df.iloc[0]
    if row["executable_lines"] != EXPECTED_EXECUTABLE_LINES:
        pytest.fail(f"Expected executable_lines={EXPECTED_EXECUTABLE_LINES}, got {row['executable_lines']}")
    if row["covered_lines"] != EXPECTED_COVERED_LINES:
        pytest.fail(f"Expected covered_lines={EXPECTED_COVERED_LINES}, got {row['covered_lines']}")


# =============================================================================
# Tests for materialize_table with coverage_functions
# =============================================================================


def test_materialize_table_writes_coverage_functions(test_ctx: TestContext) -> None:
    """Verify materialize_table writes coverage_functions rows to database."""
    # Tables already exist (created by apply_all_schemas in test context)
    # Insert test data using parameterized queries with correct schema
    test_ctx.con.execute(
        """
        INSERT INTO core.goids (goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line, created_at)
        VALUES (12345, 'urn:func', ?, ?, 'foo.py', 'python', 'function', 'foo', 1, 3, NOW())
        """,
        [test_ctx.repo, test_ctx.commit],
    )
    test_ctx.con.execute(
        """
        INSERT INTO analytics.coverage_lines (repo, commit, rel_path, line, is_executable, is_covered, hits, context_count, created_at)
        VALUES
            (?, ?, 'foo.py', 1, TRUE, TRUE, 1, 0, NOW()),
            (?, ?, 'foo.py', 2, TRUE, FALSE, 0, 0, NOW()),
            (?, ?, 'foo.py', 3, FALSE, FALSE, 0, 0, NOW())
        """,
        [
            test_ctx.repo,
            test_ctx.commit,
            test_ctx.repo,
            test_ctx.commit,
            test_ctx.repo,
            test_ctx.commit,
        ],
    )

    # Build expression
    expr = build_coverage_functions_expr(test_ctx.gateway, test_ctx.snapshot)
    if expr is None:
        pytest.fail("Expected Ibis table expression, got None")

    # Materialize
    ctx = MaterializationContext(
        gateway=test_ctx.gateway,
        snapshot=SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(),
        ),
        validate=False,
    )

    ref = materialize_table(ctx, "analytics.coverage_functions", expr)

    if ref.row_count != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_SINGLE}, got {ref.row_count}")

    # Verify data in database
    count = test_ctx.con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ?
        """,
        [test_ctx.repo, test_ctx.commit],
    ).fetchone()
    if count is None or count[0] != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected {EXPECTED_ROW_COUNT_SINGLE} row in DB, got {count}")


# =============================================================================
# Tests for deprecation warnings
# =============================================================================


def test_compute_coverage_functions_deprecation(test_ctx: TestContext) -> None:
    """Verify compute_coverage_functions emits DeprecationWarning."""
    # Tables already exist (created by apply_all_schemas in test context)
    with pytest.warns(DeprecationWarning, match="compute_coverage_functions is deprecated"):
        compute_coverage_functions(test_ctx.gateway, test_ctx.snapshot)


# =============================================================================
# Architecture guardrail tests
# =============================================================================


def test_coverage_functions_in_allowlist() -> None:
    """Verify analytics/compute/coverage/functions.py is in allowlist for backward compat.

    The deprecated compute_coverage_functions function still has direct DB writes
    for backward compatibility. Once the function is removed, the file
    should be removed from the allowlist.

    New code should use build_coverage_functions_expr with materialize_table.
    """
    if "src/codeintel/analytics/compute/coverage/functions.py" not in ALLOWLIST_IBIS_WRITE_FILES:
        pytest.fail(
            "analytics/compute/coverage/functions.py should remain in "
            "ALLOWLIST_IBIS_WRITE_FILES until deprecated function is removed"
        )


# =============================================================================
# Native module export tests
# =============================================================================


def test_native_module_exports() -> None:
    """Verify native module exports expected Hamilton nodes."""
    from codeintel.build.hamilton.native.analytics import (  # noqa: PLC0415
        coverage_functions as native_module,
    )

    expected = {"t__coverage_functions", "t__coverage_functions__compute"}
    actual = set(native_module.__all__)
    if actual != expected:
        pytest.fail(f"Expected exports {expected}, got {actual}")


def test_hamilton_nodes_have_tags() -> None:
    """Verify Hamilton nodes have proper tag decorators."""
    from codeintel.build.hamilton.native.analytics import (  # noqa: PLC0415
        coverage_functions as native_module,
    )

    for node in [native_module.t__coverage_functions, native_module.t__coverage_functions__compute]:
        if not hasattr(node, "decorate_nodes"):
            pytest.fail(f"{node.__name__} missing decorate_nodes (no @tag decorator)")


# =============================================================================
# Column count tests
# =============================================================================


def test_coverage_functions_output_column_count(test_ctx: TestContext) -> None:
    """Verify coverage_functions expression produces expected number of columns."""
    # Tables already exist (created by apply_all_schemas in test context)
    result = build_coverage_functions_expr(test_ctx.gateway, test_ctx.snapshot)
    if result is None:
        pytest.fail("Expected Ibis table expression, got None")

    col_count = len(result.columns)
    if col_count != EXPECTED_COVERAGE_FUNCTIONS_COLS:
        pytest.fail(
            f"Expected {EXPECTED_COVERAGE_FUNCTIONS_COLS} columns, got {col_count}: {result.columns}"
        )
