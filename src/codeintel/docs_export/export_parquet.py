"""Parquet exporters for the CodeIntel metadata warehouse."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from codeintel.docs_export.validate_exports import validate_files
from codeintel.services.errors import ExportError, problem

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
    "analytics.graph_validation": "graph_validation.parquet",
    "analytics.function_validation": "function_validation.parquet",
    # Risk factors
    "analytics.goid_risk_factors": "goid_risk_factors.parquet",
    "analytics.function_profile": "function_profile.parquet",
    "analytics.file_profile": "file_profile.parquet",
    "analytics.module_profile": "module_profile.parquet",
    "analytics.graph_metrics_functions": "graph_metrics_functions.parquet",
    "analytics.graph_metrics_modules": "graph_metrics_modules.parquet",
    "analytics.subsystems": "subsystems.parquet",
    "analytics.subsystem_modules": "subsystem_modules.parquet",
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
    assumes the database already scopes data to a single repo/commit. Table
    names are validated against the known dataset mapping to avoid unsafe SQL.

    Raises
    ------
    ValueError
        If the requested table is not in the allowed export mapping.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if table_name not in PARQUET_DATASETS:
        message = f"Refusing to export unknown table: {table_name}"
        raise ValueError(message)
    log.info("Exporting %s -> %s", table_name, output_path)
    rel = con.table(table_name)
    rel.write_parquet(str(output_path))


def export_all_parquet(
    con: duckdb.DuckDBPyConnection,
    document_output_dir: Path,
    *,
    validate_exports: bool = False,
    schemas: list[str] | None = None,
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
    validate_exports:
        Whether to validate selected datasets against JSON Schemas after export.
    schemas:
        Optional subset of schema names to validate; defaults to the standard export set.

    Raises
    ------
    ExportError
        If validation fails for any selected schema.
    """
    document_output_dir = document_output_dir.resolve()
    document_output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    for table_name, filename in PARQUET_DATASETS.items():
        output_path = document_output_dir / filename
        try:
            export_parquet_for_table(con, table_name, output_path)
            written.append(output_path)
        except duckdb.Error as exc:
            # Log and continue: some tables may legitimately be empty or missing
            log.warning(
                "Failed to export %s to %s: %s",
                table_name,
                output_path,
                exc,
            )

    if validate_exports:
        schema_list = schemas or [
            "function_profile",
            "file_profile",
            "module_profile",
            "call_graph_edges",
            "symbol_use_edges",
            "test_coverage_edges",
        ]
        for schema_name in schema_list:
            matching = [p for p in written if p.name.startswith(schema_name)]
            if not matching:
                continue
            exit_code = validate_files(schema_name, matching)
            if exit_code != 0:
                pd = problem(
                    code="export.validation_failed",
                    title="Export validation failed",
                    detail=f"Validation failed for schema {schema_name}",
                    extras={"schema": schema_name, "files": [str(p) for p in matching]},
                )
                raise ExportError(pd)
