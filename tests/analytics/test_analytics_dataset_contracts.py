"""Dataset contract tests for analytics insert helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from duckdb import DuckDBPyConnection

from codeintel.analytics.datasets import (
    DeleteScope,
    get_analytics_dataset_contract,
    insert_analytics_rows,
)
from codeintel.config.dataset_contract import (
    FunctionMetricsRow,
    GraphMetricsFunctionsRow,
)
from codeintel.storage.gateway import open_memory_gateway

# Alias for backward compatibility
FunctionGraphMetricsRow = GraphMetricsFunctionsRow


def _count_rows(con: DuckDBPyConnection, table: str, repo: str, commit: str) -> int:
    """
    Return row count for a repo/commit scope.

    Returns
    -------
    int
        Number of matching rows.

    Raises
    ------
    ValueError
        If the table name is not supported by this test helper.
    """
    if table == "analytics.function_metrics":
        query = "SELECT COUNT(*) FROM analytics.function_metrics WHERE repo = ? AND commit = ?"
    elif table == "analytics.graph_metrics_functions":
        query = (
            "SELECT COUNT(*) FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?"
        )
    else:
        message = f"Unsupported table for test count: {table}"
        raise ValueError(message)
    row = con.execute(query, [repo, commit]).fetchone()
    return int(row[0]) if row is not None else 0


def _assert_fk_graph_metrics_functions(con: DuckDBPyConnection) -> None:
    """Ensure graph_metrics_functions has matching GOIDs in core.goids."""
    rows = con.execute(
        """
        SELECT s.repo, s.commit, s.function_goid_h128
        FROM analytics.graph_metrics_functions s
        LEFT JOIN core.goids t
          ON (s.repo = t.repo AND s.commit = t.commit AND s.function_goid_h128 = t.goid_h128)
        WHERE t.repo IS NULL OR t.commit IS NULL OR t.goid_h128 IS NULL
        LIMIT 1
        """
    ).fetchall()
    if rows:
        pytest.fail("FK violation from analytics.graph_metrics_functions to core.goids")


def test_function_metrics_insertion_is_idempotent() -> None:
    """Inserting the same function_metrics rows twice should be idempotent."""
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    repo = "demo/repo"
    commit = "abc123"
    now = datetime.now(UTC)
    contract = get_analytics_dataset_contract(gateway, "analytics.function_metrics")
    row: FunctionMetricsRow = {
        "function_goid_h128": 1,
        "urn": "urn:demo#fn",
        "repo": repo,
        "commit": commit,
        "rel_path": "pkg/mod.py",
        "language": "python",
        "kind": "function",
        "qualname": "pkg.mod.fn",
        "start_line": 1,
        "end_line": 2,
        "loc": 2,
        "logical_loc": 2,
        "param_count": 1,
        "positional_params": 1,
        "keyword_only_params": 0,
        "has_varargs": False,
        "has_varkw": False,
        "is_async": False,
        "is_generator": False,
        "return_count": 0,
        "yield_count": 0,
        "raise_count": 0,
        "cyclomatic_complexity": 1,
        "max_nesting_depth": 0,
        "stmt_count": 1,
        "decorator_count": 0,
        "has_docstring": False,
        "complexity_bucket": "low",
        "created_at": now,
    }
    insert_analytics_rows(
        gateway,
        contract,
        [row],
        delete_scope=DeleteScope(params=[repo, commit]),
        scope=f"{repo}@{commit}",
    )
    first = _count_rows(gateway.con, contract.table_key, repo, commit)
    insert_analytics_rows(
        gateway,
        contract,
        [row],
        delete_scope=DeleteScope(params=[repo, commit]),
        scope=f"{repo}@{commit}",
    )
    second = _count_rows(gateway.con, contract.table_key, repo, commit)
    if first != 1 or second != 1:
        pytest.fail(f"Idempotency failure: first={first} second={second}")


def test_graph_metrics_functions_idempotent_and_fk_clean() -> None:
    """graph_metrics_functions insert should be idempotent and FK clean."""
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    repo = "demo/repo"
    commit = "deadbeef"
    now = datetime.now(UTC)
    gateway.core.insert_goids(
        [
            (
                10,
                "urn:demo#fn",
                repo,
                commit,
                "pkg/mod.py",
                "python",
                "function",
                "pkg.mod.fn",
                1,
                2,
                now.isoformat(),
            )
        ]
    )
    contract = get_analytics_dataset_contract(gateway, "analytics.graph_metrics_functions")
    row: FunctionGraphMetricsRow = {
        "repo": repo,
        "commit": commit,
        "function_goid_h128": 10,
        "call_fan_in": 1,
        "call_fan_out": 1,
        "call_in_degree": 1,
        "call_out_degree": 1,
        "call_pagerank": 0.1,
        "call_betweenness": 0.2,
        "call_closeness": 0.3,
        "call_cycle_member": False,
        "call_cycle_id": None,
        "call_layer": None,
        "created_at": now,
    }
    insert_analytics_rows(
        gateway,
        contract,
        [row],
        delete_scope=DeleteScope(params=[repo, commit]),
        scope=f"{repo}@{commit}",
    )
    first = _count_rows(gateway.con, contract.table_key, repo, commit)
    insert_analytics_rows(
        gateway,
        contract,
        [row],
        delete_scope=DeleteScope(params=[repo, commit]),
        scope=f"{repo}@{commit}",
    )
    second = _count_rows(gateway.con, contract.table_key, repo, commit)
    if first != 1 or second != 1:
        pytest.fail(f"Idempotency failure: first={first} second={second}")
    _assert_fk_graph_metrics_functions(gateway.con)
