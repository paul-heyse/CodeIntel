"""Verify exporters invoke macro paths by observing a sentinel macro call."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.pipeline.export.export_jsonl import export_jsonl_for_table
from codeintel.pipeline.export.export_parquet import export_parquet_for_table
from codeintel.storage.gateway import StorageGateway


def _seed_call_graph_edges(gateway: StorageGateway) -> None:
    con = gateway.con
    con.execute(
        """
        CREATE SCHEMA IF NOT EXISTS graph;
        CREATE TABLE IF NOT EXISTS graph.call_graph_edges (
            repo TEXT,
            commit TEXT,
            caller_goid_h128 BIGINT,
            callee_goid_h128 BIGINT,
            callsite_path TEXT,
            callsite_line INTEGER,
            callsite_col INTEGER,
            lang TEXT,
            call_type TEXT,
            local_name TEXT,
            prob DOUBLE,
            metadata JSON
        );
        """
    )
    con.execute(
        """
        INSERT INTO graph.call_graph_edges VALUES
        ('r', 'c', 1, 2, 'a.py', 1, 0, 'python', 'direct', 'local', 1.0, '{}')
        """
    )


@pytest.mark.smoke
def test_export_calls_macro_sentinel(tmp_path: Path, fresh_gateway: StorageGateway) -> None:
    """Export should fail without macro and succeed once macro is restored."""
    _seed_call_graph_edges(fresh_gateway)

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
