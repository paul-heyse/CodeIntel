"""Macro parity: seed minimal rows and export via normalized macros."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.pipeline.export.export_jsonl import NORMALIZED_MACROS, export_jsonl_for_table
from codeintel.storage.gateway import DuckDBConnection, StorageGateway


def _seed_minimal_rows(con: DuckDBConnection) -> None:
    # Seed function_metrics
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
    # Seed call_graph_edges
    con.execute(
        """
        INSERT INTO graph.call_graph_edges (
            repo, commit, caller_goid_h128, callee_goid_h128, callsite_path,
            callsite_line, callsite_col, language, kind, resolved_via,
            confidence, evidence_json
        ) VALUES ('demo/repo','deadbeef',1,2,'mod.py',1,0,'python','direct','local',1.0,'{}')
        """
    )


@pytest.mark.smoke
def test_macro_parity_exports_seeded_rows(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """Seed minimal rows and export via normalized macros."""
    con = fresh_gateway.con
    _seed_minimal_rows(con)

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
