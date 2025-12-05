"""Tests for export macro functionality.

This module consolidates tests for:
- Macro invocation during export (JSONL/Parquet)
- Macro parity (seeding and exporting rows)
- Strict macro enforcement

Consolidated from:
- test_export_macro_invocation.py
- test_export_macro_parity.py
- test_export_macro_strict.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.export.export_jsonl import NORMALIZED_MACROS, export_jsonl_for_table
from codeintel.export.export_parquet import export_parquet_for_table
from codeintel.storage.gateway import DuckDBConnection, StorageGateway

# =============================================================================
# Seed Helpers
# =============================================================================


def _seed_call_graph_edges_minimal(gateway: StorageGateway) -> None:
    """Seed call_graph_edges with minimal data using existing schema."""
    gateway.con.execute(
        """
        INSERT INTO graph.call_graph_edges (
            repo, commit, caller_goid_h128, callee_goid_h128, callsite_path,
            callsite_line, callsite_col, language, kind, resolved_via,
            confidence, evidence_json
        ) VALUES ('demo/repo', 'deadbeef', 1, 2, 'a.py', 1, 0, 'python', 'direct', 'local', 1.0, '{}')
        """
    )


def _seed_function_metrics_minimal(con: DuckDBConnection) -> None:
    """Seed function_metrics with minimal data."""
    con.execute(
        """
        INSERT INTO analytics.function_metrics (
            function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, loc, logical_loc, param_count, positional_params,
            keyword_only_params, has_varargs, has_varkw, is_async, is_generator,
            return_count, yield_count, raise_count, cyclomatic_complexity,
            max_nesting_depth, stmt_count, decorator_count, has_docstring,
            complexity_bucket, created_at
        )
        VALUES (
            1, 'urn:1', 'demo/repo', 'deadbeef', 'mod.py', 'python', 'function', 'pkg.mod.func',
            1, 10, 10, 8, 1, 1,
            0, FALSE, FALSE, FALSE, FALSE,
            1, 0, 0, 1,
            1, 1, 1, TRUE,
            'low', '2024-01-01T00:00:00Z'
        )
        """
    )


# =============================================================================
# Macro Invocation Tests
# =============================================================================


@pytest.mark.smoke
def test_export_fails_without_macro_succeeds_after_restore(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Export should fail without macro and succeed once macro is restored."""
    _seed_call_graph_edges_minimal(fresh_gateway)

    jsonl_out = tmp_path / "out.jsonl"
    parquet_out = tmp_path / "out.parquet"

    con = fresh_gateway.con
    con.execute("DROP MACRO IF EXISTS metadata.normalized_call_graph_edges")

    with pytest.raises(ValueError, match="No normalized macro"):
        export_jsonl_for_table(
            fresh_gateway,
            "graph.call_graph_edges",
            jsonl_out,
            require_normalized_macros=True,
        )

    con.execute(
        """
        CREATE OR REPLACE MACRO metadata.normalized_call_graph_edges(
            table_key TEXT,
            row_limit BIGINT := 9223372036854775807,
            row_offset BIGINT := 0
        ) AS TABLE
        SELECT
            repo,
            commit,
            CAST(caller_goid_h128 AS BIGINT) AS caller_goid_h128,
            CAST(callee_goid_h128 AS BIGINT) AS callee_goid_h128,
            * EXCLUDE (repo, commit, caller_goid_h128, callee_goid_h128)
        FROM metadata.dataset_rows(table_key, row_limit, row_offset)
        """
    )

    export_jsonl_for_table(fresh_gateway, "graph.call_graph_edges", jsonl_out)
    export_parquet_for_table(fresh_gateway, "graph.call_graph_edges", parquet_out)

    if not jsonl_out.exists() or not parquet_out.exists():
        pytest.fail("Expected macro-backed exports to create JSONL and Parquet outputs")


# =============================================================================
# Macro Parity Tests
# =============================================================================


@pytest.mark.smoke
def test_macro_parity_exports_seeded_rows(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Seed minimal rows and export via normalized macros."""
    con = fresh_gateway.con
    _seed_function_metrics_minimal(con)
    _seed_call_graph_edges_minimal(fresh_gateway)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        "analytics.function_metrics",
        "graph.call_graph_edges",
    ]

    for table_key in targets:
        if table_key not in NORMALIZED_MACROS:
            pytest.skip(f"{table_key} not macro-backed")
        output_path = out_dir / f"{table_key.replace('.', '_')}.jsonl"
        export_jsonl_for_table(
            fresh_gateway,
            table_key,
            output_path,
            require_normalized_macros=True,
        )
        if not output_path.exists() or output_path.stat().st_size == 0:
            pytest.fail(f"Export did not produce data for {table_key}")


# =============================================================================
# Strict Enforcement Tests
# =============================================================================


def test_require_macros_allows_macro_backed_tables(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Macro-backed datasets export successfully when enforcement is enabled."""
    output = tmp_path / "function_metrics.jsonl"
    export_jsonl_for_table(
        fresh_gateway,
        "analytics.function_metrics",
        output,
        require_normalized_macros=True,
    )
    if not output.exists():
        pytest.fail("Expected JSONL export output to be written")


def test_require_macros_rejects_dataset_rows_only_jsonl(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Dataset_rows-only tables are rejected by JSONL export when macros required."""
    with pytest.raises(ValueError, match="No normalized macro"):
        export_jsonl_for_table(
            fresh_gateway,
            "core.goids",
            tmp_path / "goids.jsonl",
            require_normalized_macros=True,
        )


def test_require_macros_rejects_dataset_rows_only_parquet(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Dataset_rows-only tables are rejected by Parquet export when macros required."""
    with pytest.raises(ValueError, match="No normalized macro"):
        export_parquet_for_table(
            fresh_gateway,
            "core.goids",
            tmp_path / "goids.parquet",
            require_normalized_macros=True,
        )
