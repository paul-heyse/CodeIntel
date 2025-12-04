"""Tests for extended CFG/DFG metrics."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.analytics.cfg_dfg import compute_cfg_metrics, compute_dfg_metrics
from tests._helpers import TestContext
from tests._helpers.builders import (
    CFGBlockRow,
    CFGEdgeRow,
    DFGEdgeRow,
    GoidRow,
    ModuleRow,
    insert_rows,
)

# Test constants
REL_PATH = "pkg/mod.py"
GOID_TEST_FUNC = 1

# Expected metrics values
EXPECTED_CFG_EXT_METRICS = (1, 1, 0, 0, 1, 2, 1, 1)
PHI_RATIO_MIN = 0.3
PHI_RATIO_MAX = 0.4


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

    Creates a CFG with:
    - Entry block
    - Body block
    - Loop header block
    - Unreachable block
    - Exit block
    - Various edge types (fallthrough, loop, back)

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
            CFGBlockRow(
                GOID_TEST_FUNC, 2, "1:block2", "loop_head", REL_PATH, 4, 4, "loop_head", "[]", 1, 2
            ),
            CFGBlockRow(
                GOID_TEST_FUNC, 3, "1:block3", "unreachable", REL_PATH, 10, 10, "body", "[]", 0, 0
            ),
            CFGBlockRow(
                GOID_TEST_FUNC, 4, "1:block4", "exit", REL_PATH, 11, 11, "exit", "[]", 1, 0
            ),
        ],
    )

    insert_rows(
        ctx.gateway,
        [
            CFGEdgeRow(GOID_TEST_FUNC, "1:block0", "1:block1", "fallthrough"),
            CFGEdgeRow(GOID_TEST_FUNC, "1:block1", "1:block2", "loop"),
            CFGEdgeRow(GOID_TEST_FUNC, "1:block2", "1:block1", "back"),
            CFGEdgeRow(GOID_TEST_FUNC, "1:block2", "1:block4", "fallthrough"),
        ],
    )


def _seed_dfg(ctx: TestContext) -> None:
    """Seed DFG edges for testing.

    Creates DFG with:
    - Data flow edge
    - Phi edge
    - Intra-block edge

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
            DFGEdgeRow(
                GOID_TEST_FUNC,
                "1:block1",
                "1:block2",
                "a",
                "a",
                "phi",
                via_phi=True,
                use_kind="phi",
            ),
            DFGEdgeRow(
                GOID_TEST_FUNC,
                "1:block1",
                "1:block1",
                "a",
                "a",
                "intra-block",
                via_phi=False,
                use_kind="intra-block",
            ),
        ],
    )


def test_cfg_metrics_ext_populates_loop_and_unreachable_counts(
    test_ctx: TestContext,
) -> None:
    """Verify extended CFG metrics capture loop headers, unreachable blocks, and edge kinds."""
    _seed_function(test_ctx)
    _seed_cfg(test_ctx)

    compute_cfg_metrics(test_ctx.gateway, repo=test_ctx.repo, commit=test_ctx.commit)

    row = test_ctx.con.execute(
        """
        SELECT unreachable_block_count, loop_header_count,
               true_edge_count, false_edge_count, back_edge_count,
               fallthrough_edge_count, loop_edge_count, entry_exit_simple_paths
        FROM analytics.cfg_function_metrics_ext
        WHERE function_goid_h128 = ?
        """,
        [GOID_TEST_FUNC],
    ).fetchone()
    base_row = test_ctx.con.execute(
        """
        SELECT cfg_block_count, cfg_edge_count
        FROM analytics.cfg_function_metrics
        WHERE function_goid_h128 = ?
        """,
        [GOID_TEST_FUNC],
    ).fetchone()
    if base_row is None:
        pytest.fail("Base CFG metrics missing; compute_cfg_metrics did not process function")
    if row != EXPECTED_CFG_EXT_METRICS:
        pytest.fail(f"Unexpected CFG ext metrics: {row}")


def test_dfg_metrics_ext_counts_use_kinds_and_paths(
    test_ctx: TestContext,
) -> None:
    """Verify extended DFG metrics capture use-kind counts and simple path totals."""
    _seed_function(test_ctx)
    _seed_dfg(test_ctx)

    compute_dfg_metrics(test_ctx.gateway, repo=test_ctx.repo, commit=test_ctx.commit)

    row = test_ctx.con.execute(
        """
        SELECT data_flow_edge_count, intra_block_edge_count,
               use_kind_phi_count, use_kind_data_flow_count, use_kind_intra_block_count,
               phi_edge_ratio, entry_exit_simple_paths
        FROM analytics.dfg_function_metrics_ext
        WHERE function_goid_h128 = ?
        """,
        [GOID_TEST_FUNC],
    ).fetchone()
    if row is None:
        pytest.fail("DFG ext metrics missing")
    (
        data_flow_edge_count,
        intra_block_edge_count,
        use_kind_phi_count,
        use_kind_data_flow_count,
        use_kind_intra_block_count,
        phi_edge_ratio,
        simple_paths,
    ) = row
    if data_flow_edge_count != 1 or intra_block_edge_count != 1:
        pytest.fail(f"Unexpected DFG edge counts: {row}")
    if use_kind_phi_count != 1 or use_kind_data_flow_count != 1 or use_kind_intra_block_count != 1:
        pytest.fail(f"Unexpected use-kind counts: {row}")
    if not (PHI_RATIO_MIN < phi_edge_ratio < PHI_RATIO_MAX):
        pytest.fail(f"Unexpected phi ratio: {phi_edge_ratio}")
    if simple_paths < 1:
        pytest.fail(f"Expected at least one simple path, got {simple_paths}")
