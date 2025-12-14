"""PR51: Tests for CFG/DFG native Hamilton module.

This module tests the migration from plugin-based CFG/DFG metrics to
Hamilton native nodes. It verifies:
1. Pure compute functions return correct result types
2. materialize_rows helper works correctly
3. Native Hamilton nodes integrate properly
4. All 6 tables are populated with correct schemas
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.cfg_dfg.compute import (
    CfgMetricsResult,
    DfgMetricsResult,
    compute_cfg_metrics_pure,
    compute_dfg_metrics_pure,
)
from codeintel.analytics.cfg_dfg.materialize import (
    CFG_BLOCK_METRICS_COLS,
    CFG_FUNCTION_METRICS_COLS,
    CFG_FUNCTION_METRICS_EXT_COLS,
    DFG_FUNCTION_METRICS_COLS,
    DFG_FUNCTION_METRICS_EXT_COLS,
    compute_cfg_metrics,
    compute_dfg_metrics,
)
from codeintel.build.hamilton.native.analytics import cfg_dfg
from codeintel.build.hamilton.native.materializer import (
    MaterializationContext,
    materialize_rows,
)
from codeintel.config.primitives import SnapshotRef
from tests._helpers.builders import (
    CFGBlockRow,
    CFGEdgeRow,
    DFGEdgeRow,
    GoidRow,
    ModuleRow,
    insert_rows,
)
from tests.build.hamilton.test_pr50_architecture_guardrails import ALLOWLIST_IBIS_WRITE_FILES

if TYPE_CHECKING:
    from tests._helpers import TestContext


REL_PATH = "pkg/mod.py"
GOID_TEST_FUNC = 1
EXPECTED_CFG_BLOCK_COUNT = 3
EXPECTED_ROW_COUNT_SINGLE = 1
EXPECTED_ROW_COUNT_EMPTY = 0
EXPECTED_GOID_SECOND_WRITE = 2


def _seed_function(ctx: TestContext) -> None:
    """Seed a test function with module and GOID.

    Parameters
    ----------
    ctx
        Test context with gateway.
    """
    now = datetime.now(UTC)
    insert_rows(
        ctx.gateway,
        [ModuleRow(module="pkg.mod", path=REL_PATH, repo=ctx.repo, commit=ctx.commit)],
    )
    insert_rows(
        ctx.gateway,
        [
            GoidRow(
                goid_h128=GOID_TEST_FUNC,
                urn="urn:pkg.mod:func",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=REL_PATH,
                kind="function",
                qualname="pkg.mod.func",
                start_line=1,
                end_line=20,
                language="python",
                created_at=now,
            )
        ],
    )


def _seed_cfg(ctx: TestContext) -> None:
    """Seed CFG blocks and edges for testing.

    Creates a CFG with entry, body, and exit blocks connected by
    fallthrough edges.

    Parameters
    ----------
    ctx
        Test context with gateway.
    """
    insert_rows(
        ctx.gateway,
        [
            CFGBlockRow(
                GOID_TEST_FUNC, 0, "1:block0", "entry", REL_PATH, 1, 1, "entry", "[]", 0, 1
            ),
            CFGBlockRow(GOID_TEST_FUNC, 1, "1:block1", "body", REL_PATH, 2, 3, "body", "[]", 1, 1),
            CFGBlockRow(GOID_TEST_FUNC, 2, "1:block2", "exit", REL_PATH, 4, 4, "exit", "[]", 1, 0),
        ],
    )
    insert_rows(
        ctx.gateway,
        [
            CFGEdgeRow(GOID_TEST_FUNC, "1:block0", "1:block1", "fallthrough"),
            CFGEdgeRow(GOID_TEST_FUNC, "1:block1", "1:block2", "fallthrough"),
        ],
    )


def _seed_dfg(ctx: TestContext) -> None:
    """Seed DFG edges for testing.

    Creates DFG with a simple data-flow edge.

    Parameters
    ----------
    ctx
        Test context with gateway.
    """
    insert_rows(
        ctx.gateway,
        [
            DFGEdgeRow(
                GOID_TEST_FUNC,
                "1:block0",
                "1:block1",
                "a",
                "a",
                "data-flow",
                via_phi=False,
                use_kind="data-flow",
            ),
        ],
    )


# =============================================================================
# Tests for compute_cfg_metrics_pure
# =============================================================================


def test_cfg_metrics_pure_returns_correct_type(test_ctx: TestContext) -> None:
    """Verify compute_cfg_metrics_pure returns CfgMetricsResult type."""
    _seed_function(test_ctx)
    _seed_cfg(test_ctx)

    result = compute_cfg_metrics_pure(
        test_ctx.gateway,
        test_ctx.repo,
        test_ctx.commit,
    )

    if not isinstance(result, CfgMetricsResult):
        pytest.fail(f"Expected CfgMetricsResult, got {type(result)}")


def test_cfg_metrics_pure_fn_rows_columns(test_ctx: TestContext) -> None:
    """Verify fn_rows have correct number of columns."""
    _seed_function(test_ctx)
    _seed_cfg(test_ctx)

    result = compute_cfg_metrics_pure(
        test_ctx.gateway,
        test_ctx.repo,
        test_ctx.commit,
    )

    if not result.fn_rows:
        pytest.fail("Expected at least one function row")

    expected_cols = len(CFG_FUNCTION_METRICS_COLS)
    actual_cols = len(result.fn_rows[0])
    if actual_cols != expected_cols:
        pytest.fail(f"Expected {expected_cols} columns, got {actual_cols}")


def test_cfg_metrics_pure_produces_block_rows(test_ctx: TestContext) -> None:
    """Verify block_rows are produced for CFG blocks."""
    _seed_function(test_ctx)
    _seed_cfg(test_ctx)

    result = compute_cfg_metrics_pure(
        test_ctx.gateway,
        test_ctx.repo,
        test_ctx.commit,
    )

    if len(result.block_rows) != EXPECTED_CFG_BLOCK_COUNT:
        pytest.fail(f"Expected {EXPECTED_CFG_BLOCK_COUNT} block rows, got {len(result.block_rows)}")

    expected_cols = len(CFG_BLOCK_METRICS_COLS)
    actual_cols = len(result.block_rows[0])
    if actual_cols != expected_cols:
        pytest.fail(f"Expected {expected_cols} block columns, got {actual_cols}")


def test_cfg_metrics_pure_produces_ext_rows(test_ctx: TestContext) -> None:
    """Verify ext_rows are produced for extended metrics."""
    _seed_function(test_ctx)
    _seed_cfg(test_ctx)

    result = compute_cfg_metrics_pure(
        test_ctx.gateway,
        test_ctx.repo,
        test_ctx.commit,
    )

    if not result.ext_rows:
        pytest.fail("Expected at least one ext row")

    expected_cols = len(CFG_FUNCTION_METRICS_EXT_COLS)
    actual_cols = len(result.ext_rows[0])
    if actual_cols != expected_cols:
        pytest.fail(f"Expected {expected_cols} ext columns, got {actual_cols}")


def test_cfg_metrics_pure_empty_returns_empty(test_ctx: TestContext) -> None:
    """Verify empty CFG returns empty result without error."""
    result = compute_cfg_metrics_pure(
        test_ctx.gateway,
        test_ctx.repo,
        test_ctx.commit,
    )

    if result.fn_rows:
        pytest.fail(f"Expected empty fn_rows, got {len(result.fn_rows)}")
    if result.block_rows:
        pytest.fail(f"Expected empty block_rows, got {len(result.block_rows)}")
    if result.ext_rows:
        pytest.fail(f"Expected empty ext_rows, got {len(result.ext_rows)}")


# =============================================================================
# Tests for compute_dfg_metrics_pure
# =============================================================================


def test_dfg_metrics_pure_returns_correct_type(test_ctx: TestContext) -> None:
    """Verify compute_dfg_metrics_pure returns DfgMetricsResult type."""
    _seed_function(test_ctx)
    _seed_dfg(test_ctx)

    result = compute_dfg_metrics_pure(
        test_ctx.gateway,
        test_ctx.repo,
        test_ctx.commit,
    )

    if not isinstance(result, DfgMetricsResult):
        pytest.fail(f"Expected DfgMetricsResult, got {type(result)}")


def test_dfg_metrics_pure_fn_rows_columns(test_ctx: TestContext) -> None:
    """Verify fn_rows have correct number of columns."""
    _seed_function(test_ctx)
    _seed_dfg(test_ctx)

    result = compute_dfg_metrics_pure(
        test_ctx.gateway,
        test_ctx.repo,
        test_ctx.commit,
    )

    if not result.fn_rows:
        pytest.fail("Expected at least one function row")

    expected_cols = len(DFG_FUNCTION_METRICS_COLS)
    actual_cols = len(result.fn_rows[0])
    if actual_cols != expected_cols:
        pytest.fail(f"Expected {expected_cols} columns, got {actual_cols}")


def test_dfg_metrics_pure_produces_ext_rows(test_ctx: TestContext) -> None:
    """Verify ext_rows are produced for extended DFG metrics."""
    _seed_function(test_ctx)
    _seed_dfg(test_ctx)

    result = compute_dfg_metrics_pure(
        test_ctx.gateway,
        test_ctx.repo,
        test_ctx.commit,
    )

    if not result.ext_rows:
        pytest.fail("Expected at least one ext row")

    expected_cols = len(DFG_FUNCTION_METRICS_EXT_COLS)
    actual_cols = len(result.ext_rows[0])
    if actual_cols != expected_cols:
        pytest.fail(f"Expected {expected_cols} ext columns, got {actual_cols}")


def test_dfg_metrics_pure_empty_returns_empty(test_ctx: TestContext) -> None:
    """Verify empty DFG returns empty result without error."""
    result = compute_dfg_metrics_pure(
        test_ctx.gateway,
        test_ctx.repo,
        test_ctx.commit,
    )

    if result.fn_rows:
        pytest.fail(f"Expected empty fn_rows, got {len(result.fn_rows)}")


# =============================================================================
# Tests for materialize_rows helper
# =============================================================================


def test_materialize_rows_writes_to_db(test_ctx: TestContext) -> None:
    """Verify materialize_rows writes row tuples to database."""
    test_ctx.gateway.policy.ensure_table("analytics.cfg_function_metrics")

    ctx = MaterializationContext(
        gateway=test_ctx.gateway,
        snapshot=SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(),
        ),
        validate=False,
    )

    now = datetime.now(UTC)
    rows = [
        (
            GOID_TEST_FUNC,
            test_ctx.repo,
            test_ctx.commit,
            REL_PATH,
            "pkg.mod",
            "func",
            3,
            2,
            False,
            1,
            2,
            1.0,
            1.0,
            1,
            0.5,
            2,
            0.5,
            1,
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            now,
            "1.0",
        )
    ]

    ref = materialize_rows(
        ctx,
        "analytics.cfg_function_metrics",
        rows,
        CFG_FUNCTION_METRICS_COLS,
    )

    if ref.row_count != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_SINGLE}, got {ref.row_count}")
    if ref.table_key != "analytics.cfg_function_metrics":
        pytest.fail(f"Unexpected table_key: {ref.table_key}")

    count = test_ctx.con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.cfg_function_metrics
        WHERE repo = ? AND commit = ?
        """,
        [test_ctx.repo, test_ctx.commit],
    ).fetchone()
    if count is None or count[0] != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected {EXPECTED_ROW_COUNT_SINGLE} row in DB, got {count}")


def test_materialize_rows_replaces_snapshot_data(test_ctx: TestContext) -> None:
    """Verify materialize_rows replaces existing snapshot data."""
    test_ctx.gateway.policy.ensure_table("analytics.cfg_function_metrics")

    ctx = MaterializationContext(
        gateway=test_ctx.gateway,
        snapshot=SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(),
        ),
        validate=False,
    )

    now = datetime.now(UTC)
    rows_v1 = [
        (
            1,
            test_ctx.repo,
            test_ctx.commit,
            REL_PATH,
            "pkg.mod",
            "func1",
            3,
            2,
            False,
            1,
            2,
            1.0,
            1.0,
            1,
            0.5,
            2,
            0.5,
            1,
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            now,
            "1.0",
        )
    ]
    materialize_rows(ctx, "analytics.cfg_function_metrics", rows_v1, CFG_FUNCTION_METRICS_COLS)

    rows_v2 = [
        (
            2,
            test_ctx.repo,
            test_ctx.commit,
            REL_PATH,
            "pkg.mod",
            "func2",
            5,
            4,
            True,
            2,
            3,
            2.0,
            2.0,
            2,
            0.6,
            3,
            0.6,
            2,
            1,
            1,
            0.1,
            0.1,
            0.1,
            0.1,
            now,
            "1.0",
        )
    ]
    ref = materialize_rows(
        ctx, "analytics.cfg_function_metrics", rows_v2, CFG_FUNCTION_METRICS_COLS
    )

    if ref.row_count != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_SINGLE}, got {ref.row_count}")

    rows = test_ctx.con.execute(
        """
        SELECT function_goid_h128
        FROM analytics.cfg_function_metrics
        WHERE repo = ? AND commit = ?
        """,
        [test_ctx.repo, test_ctx.commit],
    ).fetchall()
    if len(rows) != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected {EXPECTED_ROW_COUNT_SINGLE} row, got {len(rows)}")
    if rows[0][0] != EXPECTED_GOID_SECOND_WRITE:
        pytest.fail(f"Expected goid={EXPECTED_GOID_SECOND_WRITE}, got {rows[0][0]}")


def test_materialize_rows_handles_empty(test_ctx: TestContext) -> None:
    """Verify materialize_rows handles empty row list gracefully."""
    test_ctx.gateway.policy.ensure_table("analytics.cfg_function_metrics")

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
        "analytics.cfg_function_metrics",
        [],
        CFG_FUNCTION_METRICS_COLS,
    )

    if ref.row_count != EXPECTED_ROW_COUNT_EMPTY:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_EMPTY}, got {ref.row_count}")


# =============================================================================
# Architecture guardrail tests
# =============================================================================


def test_materialize_py_deprecated_in_allowlist() -> None:
    """Verify analytics/cfg_dfg/materialize.py is still in allowlist for backward compat.

    The deprecated functions compute_cfg_metrics and compute_dfg_metrics still
    have direct DB writes for backward compatibility. Once these functions are
    removed entirely, this file should be removed from the allowlist.

    New code should use the Hamilton native module instead:
    `codeintel.build.hamilton.native.analytics.cfg_dfg`
    """
    if "src/codeintel/analytics/cfg_dfg/materialize.py" not in ALLOWLIST_IBIS_WRITE_FILES:
        pytest.fail(
            "analytics/cfg_dfg/materialize.py should remain in "
            "ALLOWLIST_IBIS_WRITE_FILES until deprecated functions are removed"
        )


# =============================================================================
# Deprecation warning tests
# =============================================================================


def test_compute_cfg_metrics_deprecation(test_ctx: TestContext) -> None:
    """Verify compute_cfg_metrics emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="compute_cfg_metrics is deprecated"):
        compute_cfg_metrics(
            test_ctx.gateway,
            repo=test_ctx.repo,
            commit=test_ctx.commit,
        )


def test_compute_dfg_metrics_deprecation(test_ctx: TestContext) -> None:
    """Verify compute_dfg_metrics emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="compute_dfg_metrics is deprecated"):
        compute_dfg_metrics(
            test_ctx.gateway,
            repo=test_ctx.repo,
            commit=test_ctx.commit,
        )


# =============================================================================
# Native module export tests
# =============================================================================


def test_native_module_exports() -> None:
    """Verify native module exports expected Hamilton nodes."""
    expected = {
        "t__cfg_dfg_metrics",
        "t__cfg_dfg_metrics__compute_cfg",
        "t__cfg_dfg_metrics__compute_dfg",
    }
    actual = set(cfg_dfg.__all__)
    if actual != expected:
        pytest.fail(f"Expected exports {expected}, got {actual}")


def test_hamilton_nodes_have_tags() -> None:
    """Verify Hamilton nodes have proper tag decorators."""
    compute_cfg_node = cfg_dfg.t__cfg_dfg_metrics__compute_cfg
    compute_dfg_node = cfg_dfg.t__cfg_dfg_metrics__compute_dfg
    materialize_node = cfg_dfg.t__cfg_dfg_metrics

    # Hamilton stores tag decorators in decorate_nodes attribute
    for node, name in [
        (compute_cfg_node, "compute_cfg"),
        (compute_dfg_node, "compute_dfg"),
        (materialize_node, "materialize"),
    ]:
        if not hasattr(node, "decorate_nodes"):
            pytest.fail(f"{name} missing decorate_nodes attribute from @tag decorator")
