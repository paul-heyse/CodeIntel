"""Tests for extended CFG/DFG metrics."""

from __future__ import annotations

import pytest

from codeintel.analytics.cfg_dfg import compute_cfg_metrics, compute_dfg_metrics
from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import (
    CFGBlockRow,
    CFGEdgeRow,
    DFGEdgeRow,
    GoidRow,
    ModuleRow,
    insert_cfg_blocks,
    insert_cfg_edges,
    insert_dfg_edges,
    insert_goids,
    insert_modules,
)

REPO = "demo/repo"
COMMIT = "abc123"
REL_PATH = "pkg/mod.py"


def _seed_function(gateway: StorageGateway) -> None:
    insert_modules(
        gateway,
        [ModuleRow(module="pkg.mod", path=REL_PATH, repo=REPO, commit=COMMIT)],
    )
    insert_goids(
        gateway,
        [
            GoidRow(
                goid_h128=1,
                urn="urn:pkg.mod:func",
                repo=REPO,
                commit=COMMIT,
                rel_path=REL_PATH,
                kind="function",
                qualname="pkg.mod:func",
                start_line=1,
                end_line=20,
            )
        ],
    )


def _seed_cfg(gateway: StorageGateway) -> None:
    insert_cfg_blocks(
        gateway,
        [
            CFGBlockRow(1, 0, "1:block0", "entry", REL_PATH, 1, 1, "entry", "[]", 0, 1),
            CFGBlockRow(1, 1, "1:block1", "body", REL_PATH, 2, 3, "body", "[]", 1, 1),
            CFGBlockRow(1, 2, "1:block2", "loop_head", REL_PATH, 4, 4, "loop_head", "[]", 1, 2),
            CFGBlockRow(1, 3, "1:block3", "unreachable", REL_PATH, 10, 10, "body", "[]", 0, 0),
            CFGBlockRow(1, 4, "1:block4", "exit", REL_PATH, 11, 11, "exit", "[]", 1, 0),
        ],
    )

    insert_cfg_edges(
        gateway,
        [
            CFGEdgeRow(1, "1:block0", "1:block1", "fallthrough"),
            CFGEdgeRow(1, "1:block1", "1:block2", "loop"),
            CFGEdgeRow(1, "1:block2", "1:block1", "back"),
            CFGEdgeRow(1, "1:block2", "1:block4", "fallthrough"),
        ],
    )


def _seed_dfg(gateway: StorageGateway) -> None:
    insert_dfg_edges(
        gateway,
        [
            DFGEdgeRow(
                1,
                "1:block0",
                "1:block1",
                "a",
                "a",
                "data-flow",
                via_phi=False,
                use_kind="data-flow",
            ),
            DFGEdgeRow(
                1,
                "1:block1",
                "1:block2",
                "a",
                "a",
                "phi",
                via_phi=True,
                use_kind="phi",
            ),
            DFGEdgeRow(
                1,
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
    fresh_gateway: StorageGateway,
) -> None:
    """Extended CFG metrics capture loop headers, unreachable blocks, and edge kinds."""
    _seed_function(fresh_gateway)
    _seed_cfg(fresh_gateway)

    compute_cfg_metrics(fresh_gateway, repo=REPO, commit=COMMIT)

    row = fresh_gateway.con.execute(
        """
        SELECT unreachable_block_count, loop_header_count,
               true_edge_count, false_edge_count, back_edge_count,
               fallthrough_edge_count, loop_edge_count, entry_exit_simple_paths
        FROM analytics.cfg_function_metrics_ext
        WHERE function_goid_h128 = ?
        """,
        [1],
    ).fetchone()
    base_row = fresh_gateway.con.execute(
        """
        SELECT cfg_block_count, cfg_edge_count
        FROM analytics.cfg_function_metrics
        WHERE function_goid_h128 = ?
        """,
        [1],
    ).fetchone()
    if base_row is None:
        pytest.fail("Base CFG metrics missing; compute_cfg_metrics did not process function")
    if row != (1, 1, 0, 0, 1, 2, 1, 1):
        pytest.fail(f"Unexpected CFG ext metrics: {row}")


def test_dfg_metrics_ext_counts_use_kinds_and_paths(
    fresh_gateway: StorageGateway,
) -> None:
    """Extended DFG metrics capture use-kind counts and simple path totals."""
    _seed_function(fresh_gateway)
    _seed_dfg(fresh_gateway)

    compute_dfg_metrics(fresh_gateway, repo=REPO, commit=COMMIT)

    row = fresh_gateway.con.execute(
        """
        SELECT data_flow_edge_count, intra_block_edge_count,
               use_kind_phi_count, use_kind_data_flow_count, use_kind_intra_block_count,
               phi_edge_ratio, entry_exit_simple_paths
        FROM analytics.dfg_function_metrics_ext
        WHERE function_goid_h128 = ?
        """,
        [1],
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
    phi_ratio_min = 0.3
    phi_ratio_max = 0.4
    if data_flow_edge_count != 1 or intra_block_edge_count != 1:
        pytest.fail(f"Unexpected DFG edge counts: {row}")
    if use_kind_phi_count != 1 or use_kind_data_flow_count != 1 or use_kind_intra_block_count != 1:
        pytest.fail(f"Unexpected use-kind counts: {row}")
    if not (phi_ratio_min < phi_edge_ratio < phi_ratio_max):
        pytest.fail(f"Unexpected phi ratio: {phi_edge_ratio}")
    if simple_paths < 1:
        pytest.fail(f"Expected at least one simple path, got {simple_paths}")
