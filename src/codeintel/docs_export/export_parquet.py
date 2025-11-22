"""Parquet exporters for the CodeIntel metadata warehouse."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)


# Map DuckDB tables -> Document Output parquet basenames
PARQUET_DATASETS: dict[str, str] = {
    # GOIDs / crosswalk
    "core.goids": "goids.parquet",
    "core.goid_crosswalk": "goid_crosswalk.parquet",
    # Call graph
    "graph.call_graph_nodes": "call_graph_nodes.parquet",
    "graph.call_graph_edges": "call_graph_edges.parquet",
    # CFG / DFG
    "graph.cfg_blocks": "cfg_blocks.parquet",
    "graph.cfg_edges": "cfg_edges.parquet",
    "graph.dfg_edges": "dfg_edges.parquet",
    # Import / symbol uses
    "graph.import_graph_edges": "import_graph_edges.parquet",
    "graph.symbol_use_edges": "symbol_use_edges.parquet",
    # AST / CST
    "core.ast_nodes": "ast_nodes.parquet",
    "core.ast_metrics": "ast_metrics.parquet",
    "core.cst_nodes": "cst_nodes.parquet",
    "core.docstrings": "docstrings.parquet",
    # Modules / config / diagnostics
    "core.modules": "modules.parquet",
    "analytics.config_values": "config_values.parquet",
    "analytics.static_diagnostics": "static_diagnostics.parquet",
    # AST analytics / typing
    "analytics.hotspots": "hotspots.parquet",
    "analytics.typedness": "typedness.parquet",
    # Function analytics
    "analytics.function_metrics": "function_metrics.parquet",
    "analytics.function_types": "function_types.parquet",
    # Coverage + tests
    "analytics.coverage_lines": "coverage_lines.parquet",
    "analytics.coverage_functions": "coverage_functions.parquet",
    "analytics.test_catalog": "test_catalog.parquet",
    "analytics.test_coverage_edges": "test_coverage_edges.parquet",
    # Risk factors
    "analytics.goid_risk_factors": "goid_risk_factors.parquet",
}


def export_parquet_for_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    output_path: Path,
) -> None:
    """
    Export a single DuckDB table to Parquet.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Live DuckDB connection.
    table_name : str
        Fully qualified table name (schema.table) to export.
    output_path : Path
        Destination path for the Parquet file.

    Notes
    -----
    Uses `COPY (SELECT * FROM <table>) TO <path> (FORMAT PARQUET)`. The export
    assumes the database already scopes data to a single repo/commit.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sql = f"COPY (SELECT * FROM {table_name}) TO ? (FORMAT PARQUET);"
    log.info("Exporting %s -> %s", table_name, output_path)
    con.execute(sql, [str(output_path)])


def export_all_parquet(
    con: duckdb.DuckDBPyConnection,
    document_output_dir: Path,
) -> None:
    """
    Export all known datasets to Parquet files under `Document Output/`.

    Parameters
    ----------
    con:
        DuckDB connection with all tables populated.
    document_output_dir:
        Path to the `Document Output/` directory (will be created
        if it does not exist).
    """
    document_output_dir = document_output_dir.resolve()
    document_output_dir.mkdir(parents=True, exist_ok=True)

    for table_name, filename in PARQUET_DATASETS.items():
        output_path = document_output_dir / filename
        try:
            export_parquet_for_table(con, table_name, output_path)
        except duckdb.Error as exc:
            # Log and continue: some tables may legitimately be empty or missing
            log.warning(
                "Failed to export %s to %s: %s",
                table_name,
                output_path,
                exc,
            )
