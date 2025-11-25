"""Tests for extended CFG/DFG metrics."""

from __future__ import annotations

from decimal import Decimal

import pytest

from codeintel.analytics.cfg_dfg_metrics import compute_cfg_metrics, compute_dfg_metrics
from codeintel.ingestion.common import run_batch
from codeintel.storage.gateway import StorageGateway

REPO = "demo/repo"
COMMIT = "abc123"
REL_PATH = "pkg/mod.py"


def _seed_function(gateway: StorageGateway) -> None:
    gateway.con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, 'python', '[]', '[]')
        """,
        ["pkg.mod", REL_PATH, REPO, COMMIT],
    )
    gateway.con.execute(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, created_at
        )
        VALUES (?, ?, ?, ?, ?, 'python', 'function', ?, 1, 20, CURRENT_TIMESTAMP)
        """,
        [Decimal(1), "urn:pkg.mod:func", REPO, COMMIT, REL_PATH, "pkg.mod:func"],
    )


def _seed_cfg(gateway: StorageGateway) -> None:
    cfg_blocks = [
        (1, 0, "1:block0", "entry", REL_PATH, 1, 1, "entry", "[]", 0, 1),
        (1, 1, "1:block1", "body", REL_PATH, 2, 3, "body", "[]", 1, 1),
        (1, 2, "1:block2", "loop_head", REL_PATH, 4, 4, "loop_head", "[]", 1, 2),
        (1, 3, "1:block3", "unreachable", REL_PATH, 10, 10, "body", "[]", 0, 0),
        (1, 4, "1:block4", "exit", REL_PATH, 11, 11, "exit", "[]", 1, 0),
    ]
    run_batch(gateway, "graph.cfg_blocks", cfg_blocks, delete_params=[], scope="cfg_blocks")

    cfg_edges = [
        (1, "1:block0", "1:block1", "fallthrough"),
        (1, "1:block1", "1:block2", "loop"),
        (1, "1:block2", "1:block1", "back"),
        (1, "1:block2", "1:block4", "fallthrough"),
    ]
    run_batch(gateway, "graph.cfg_edges", cfg_edges, delete_params=[], scope="cfg_edges")


def _seed_dfg(gateway: StorageGateway) -> None:
    dfg_edges = [
        (1, "1:block0", "1:block1", "a", "a", "data-flow", False, "data-flow"),
        (1, "1:block1", "1:block2", "a", "a", "phi", True, "phi"),
        (1, "1:block1", "1:block1", "a", "a", "intra-block", False, "intra-block"),
    ]
    run_batch(gateway, "graph.dfg_edges", dfg_edges, delete_params=[], scope="dfg_edges")


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
        [Decimal(1)],
    ).fetchone()
    base_row = fresh_gateway.con.execute(
        """
        SELECT cfg_block_count, cfg_edge_count
        FROM analytics.cfg_function_metrics
        WHERE function_goid_h128 = ?
        """,
        [Decimal(1)],
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
        [Decimal(1)],
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
