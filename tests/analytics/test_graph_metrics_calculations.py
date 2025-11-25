"""Graph metrics integration tests covering function and module computations."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.graph_metrics import compute_graph_metrics
from codeintel.config.models import GraphMetricsConfig
from tests._helpers.fixtures import (
    graph_metrics_ready_gateway,
    seed_function_graph_cycle,
    seed_module_graph_inputs,
)

REPO = "demo/repo"
COMMIT = "abc123"
REL_PATH = "pkg/mod.py"
MODULE_A = "pkg.mod_a"
MODULE_B = "pkg.mod_b"


def test_compute_function_graph_metrics_counts_and_cycles(tmp_path: Path) -> None:
    """Compute function graph metrics with cycles and aggregated edge counts."""
    ctx = graph_metrics_ready_gateway(
        tmp_path / "graph_metrics",
        repo=REPO,
        commit=COMMIT,
        include_symbol_edges=False,
        run_metrics=False,
        build_callgraph_enabled=False,
        file_backed=False,
    )
    seed_function_graph_cycle(ctx.gateway, repo=REPO, commit=COMMIT, rel_path=REL_PATH)

    cfg = GraphMetricsConfig.from_paths(repo=REPO, commit=COMMIT)
    compute_graph_metrics(ctx.gateway, cfg)

    row = ctx.gateway.con.execute(
        """
        SELECT call_fan_in, call_fan_out, call_in_degree, call_out_degree,
               call_cycle_member
        FROM analytics.graph_metrics_functions
        WHERE function_goid_h128 = 2
        """
    ).fetchone()
    if row != (1, 1, 2, 1, True):
        pytest.fail(f"Unexpected function metrics row: {row}")
    ctx.close()


def test_compute_module_graph_metrics_with_symbol_coupling(tmp_path: Path) -> None:
    """Compute module graph metrics including symbol coupling fan counts."""
    ctx = graph_metrics_ready_gateway(
        tmp_path / "graph_metrics_mod",
        repo=REPO,
        commit=COMMIT,
        include_symbol_edges=False,
        run_metrics=False,
        build_callgraph_enabled=False,
        file_backed=False,
    )
    seed_module_graph_inputs(
        ctx.gateway,
        repo=REPO,
        commit=COMMIT,
        module_a=MODULE_A,
        module_b=MODULE_B,
    )

    cfg = GraphMetricsConfig.from_paths(repo=REPO, commit=COMMIT)
    compute_graph_metrics(ctx.gateway, cfg)

    row = ctx.gateway.con.execute(
        """
        SELECT import_fan_in, import_fan_out, symbol_fan_in, symbol_fan_out, import_cycle_member
        FROM analytics.graph_metrics_modules
        WHERE module = ?
        """,
        [MODULE_A],
    ).fetchone()
    if row != (0, 1, 0, 1, False):
        pytest.fail(f"Unexpected module metrics row: {row}")
    ctx.close()


def test_graph_metrics_ready_gateway_smoke(tmp_path: Path) -> None:
    """End-to-end helper produces graph metrics rows."""
    ctx = graph_metrics_ready_gateway(tmp_path / "gm_smoke", repo=REPO, commit=COMMIT)
    con = ctx.gateway.con
    row = con.execute(
        """
        SELECT COUNT(*) FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?
        """,
        [REPO, COMMIT],
    ).fetchone()
    if row is None or int(row[0]) <= 0:
        pytest.fail("graph_metrics_ready_gateway did not produce function metrics")
    ctx.close()
