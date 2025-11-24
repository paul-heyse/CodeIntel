"""Graph metrics integration tests covering function and module computations."""

from __future__ import annotations

from decimal import Decimal

import duckdb
import pytest

from codeintel.analytics.graph_metrics import compute_graph_metrics
from codeintel.config.models import GraphMetricsConfig
from codeintel.storage.schemas import apply_all_schemas

REPO = "demo/repo"
COMMIT = "abc123"
REL_PATH = "pkg/mod.py"
MODULE_A = "pkg.mod_a"
MODULE_B = "pkg.mod_b"


def _setup_db() -> duckdb.DuckDBPyConnection:
    """
    Create an in-memory DuckDB instance with schemas applied.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Connected in-memory database ready for inserts.
    """
    con = duckdb.connect(database=":memory:")
    apply_all_schemas(con)
    return con


def test_compute_function_graph_metrics_counts_and_cycles() -> None:
    """Compute function graph metrics with cycles and aggregated edge counts."""
    con = _setup_db()

    con.execute(
        """
        INSERT INTO graph.call_graph_edges (
            caller_goid_h128, callee_goid_h128, callsite_path, callsite_line,
            callsite_col, language, kind, resolved_via, confidence, evidence_json
        ) VALUES
            (?, ?, ?, 1, 1, 'python', 'direct', 'local_name', 1.0, '{}'),
            (?, ?, ?, 2, 2, 'python', 'direct', 'local_name', 1.0, '{}')
        """,
        [1, 2, REL_PATH, 1, 2, REL_PATH],
    )
    con.execute(
        """
        INSERT INTO graph.call_graph_edges (
            caller_goid_h128, callee_goid_h128, callsite_path, callsite_line,
            callsite_col, language, kind, resolved_via, confidence, evidence_json
        ) VALUES (?, ?, ?, 3, 1, 'python', 'direct', 'local_name', 1.0, '{}')
        """,
        [Decimal("2"), Decimal("1"), REL_PATH],
    )
    con.execute(
        """
        INSERT INTO graph.call_graph_nodes (goid_h128, language, kind, arity, is_public, rel_path)
        VALUES
            (1, 'python', 'function', 0, TRUE, ?),
            (2, 'python', 'function', 0, FALSE, ?)
        """,
        [REL_PATH, REL_PATH],
    )

    cfg = GraphMetricsConfig.from_paths(repo=REPO, commit=COMMIT)
    compute_graph_metrics(con, cfg)

    row = con.execute(
        """
        SELECT call_fan_in, call_fan_out, call_in_degree, call_out_degree,
               call_cycle_member
        FROM analytics.graph_metrics_functions
        WHERE function_goid_h128 = 2
        """
    ).fetchone()
    if row != (1, 1, 2, 1, True):
        pytest.fail(f"Unexpected function metrics row: {row}")


def test_compute_module_graph_metrics_with_symbol_coupling() -> None:
    """Compute module graph metrics including symbol coupling fan counts."""
    con = _setup_db()

    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES
            (?, ?, ?, ?, 'python', '["api"]', '[]'),
            (?, ?, ?, ?, 'python', '["core"]', '[]')
        """,
        [
            MODULE_A,
            "pkg/mod_a.py",
            REPO,
            COMMIT,
            MODULE_B,
            "pkg/mod_b.py",
            REPO,
            COMMIT,
        ],
    )
    con.execute(
        """
        INSERT INTO graph.import_graph_edges (
            src_module, dst_module, src_fan_out, dst_fan_in, cycle_group
        )
        VALUES (?, ?, 1, 1, 0)
        """,
        [MODULE_A, MODULE_B],
    )
    con.execute(
        """
        INSERT INTO graph.symbol_use_edges (symbol, def_path, use_path, same_file, same_module)
        VALUES ('sym', 'pkg/mod_b.py', 'pkg/mod_a.py', FALSE, FALSE)
        """
    )

    cfg = GraphMetricsConfig.from_paths(repo=REPO, commit=COMMIT)
    compute_graph_metrics(con, cfg)

    row = con.execute(
        """
        SELECT import_fan_in, import_fan_out, symbol_fan_in, symbol_fan_out, import_cycle_member
        FROM analytics.graph_metrics_modules
        WHERE module = ?
        """,
        [MODULE_A],
    ).fetchone()
    if row != (0, 1, 0, 1, False):
        pytest.fail(f"Unexpected module metrics row: {row}")
