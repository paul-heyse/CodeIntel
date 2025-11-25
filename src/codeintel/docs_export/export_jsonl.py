"""JSON/JSONL exporters for the CodeIntel metadata warehouse."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import duckdb

from codeintel.docs_export import DEFAULT_VALIDATION_SCHEMAS
from codeintel.docs_export.validate_exports import validate_files
from codeintel.services.errors import ExportError, problem
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


# Map DuckDB tables -> Document Output JSONL basenames
JSONL_DATASETS: dict[str, str] = {
    # GOIDs / crosswalk
    "core.goids": "goids.jsonl",
    "core.goid_crosswalk": "goid_crosswalk.jsonl",
    # Call graph
    "graph.call_graph_nodes": "call_graph_nodes.jsonl",
    "graph.call_graph_edges": "call_graph_edges.jsonl",
    # CFG / DFG
    "graph.cfg_blocks": "cfg_blocks.jsonl",
    "graph.cfg_edges": "cfg_edges.jsonl",
    "graph.dfg_edges": "dfg_edges.jsonl",
    # Import / symbol uses
    "graph.import_graph_edges": "import_graph_edges.jsonl",
    "graph.symbol_use_edges": "symbol_use_edges.jsonl",
    # AST / CST
    "core.ast_nodes": "ast_nodes.jsonl",
    "core.ast_metrics": "ast_metrics.jsonl",
    "core.cst_nodes": "cst_nodes.jsonl",
    "core.docstrings": "docstrings.jsonl",
    # Modules / config / diagnostics
    "core.modules": "modules.jsonl",
    "analytics.config_values": "config_values.jsonl",
    "analytics.data_models": "data_models.jsonl",
    "analytics.data_model_fields": "data_model_fields.jsonl",
    "analytics.data_model_relationships": "data_model_relationships.jsonl",
    "analytics.data_model_usage": "data_model_usage.jsonl",
    "analytics.config_data_flow": "config_data_flow.jsonl",
    "analytics.static_diagnostics": "static_diagnostics.jsonl",
    # AST analytics / typing
    "analytics.hotspots": "hotspots.jsonl",
    "analytics.typedness": "typedness.jsonl",
    # Function analytics
    "analytics.function_metrics": "function_metrics.jsonl",
    "analytics.function_types": "function_types.jsonl",
    "analytics.function_effects": "function_effects.jsonl",
    "analytics.function_contracts": "function_contracts.jsonl",
    "analytics.semantic_roles_functions": "semantic_roles_functions.jsonl",
    "analytics.semantic_roles_modules": "semantic_roles_modules.jsonl",
    # Coverage + tests
    "analytics.coverage_lines": "coverage_lines.jsonl",
    "analytics.coverage_functions": "coverage_functions.jsonl",
    "analytics.test_catalog": "test_catalog.jsonl",
    "analytics.test_coverage_edges": "test_coverage_edges.jsonl",
    "analytics.entrypoints": "entrypoints.jsonl",
    "analytics.entrypoint_tests": "entrypoint_tests.jsonl",
    "analytics.external_dependencies": "external_dependencies.jsonl",
    "analytics.external_dependency_calls": "external_dependency_calls.jsonl",
    "analytics.graph_validation": "graph_validation.jsonl",
    "analytics.function_validation": "function_validation.jsonl",
    # Risk factors
    "analytics.goid_risk_factors": "goid_risk_factors.jsonl",
    "analytics.function_profile": "function_profile.jsonl",
    "analytics.function_history": "function_history.jsonl",
    "analytics.history_timeseries": "history_timeseries.jsonl",
    "analytics.file_profile": "file_profile.jsonl",
    "analytics.module_profile": "module_profile.jsonl",
    "analytics.graph_metrics_functions": "graph_metrics_functions.jsonl",
    "analytics.graph_metrics_functions_ext": "graph_metrics_functions_ext.jsonl",
    "analytics.graph_metrics_modules": "graph_metrics_modules.jsonl",
    "analytics.graph_metrics_modules_ext": "graph_metrics_modules_ext.jsonl",
    "analytics.subsystem_graph_metrics": "subsystem_graph_metrics.jsonl",
    "analytics.symbol_graph_metrics_modules": "symbol_graph_metrics_modules.jsonl",
    "analytics.symbol_graph_metrics_functions": "symbol_graph_metrics_functions.jsonl",
    "analytics.config_graph_metrics_keys": "config_graph_metrics_keys.jsonl",
    "analytics.config_graph_metrics_modules": "config_graph_metrics_modules.jsonl",
    "analytics.config_projection_key_edges": "config_projection_key_edges.jsonl",
    "analytics.config_projection_module_edges": "config_projection_module_edges.jsonl",
    "analytics.subsystem_agreement": "subsystem_agreement.jsonl",
    "analytics.graph_stats": "graph_stats.jsonl",
    "analytics.test_graph_metrics_tests": "test_graph_metrics_tests.jsonl",
    "analytics.test_graph_metrics_functions": "test_graph_metrics_functions.jsonl",
    "analytics.test_profile": "test_profile.jsonl",
    "analytics.behavioral_coverage": "behavioral_coverage.jsonl",
    "analytics.cfg_block_metrics": "cfg_block_metrics.jsonl",
    "analytics.cfg_function_metrics": "cfg_function_metrics.jsonl",
    "analytics.dfg_block_metrics": "dfg_block_metrics.jsonl",
    "analytics.dfg_function_metrics": "dfg_function_metrics.jsonl",
    "analytics.subsystems": "subsystems.jsonl",
    "analytics.subsystem_modules": "subsystem_modules.jsonl",
}


def _default_repo_commit(con: duckdb.DuckDBPyConnection) -> tuple[str, str]:
    row = con.execute("SELECT repo, commit FROM core.repo_map LIMIT 1").fetchone()
    if row is None:
        return "", ""
    repo, commit = row
    return str(repo), str(commit)


def _normalized_relation(
    con: duckdb.DuckDBPyConnection, table_name: str
) -> duckdb.DuckDBPyRelation:
    if table_name == "graph.call_graph_edges":
        return con.sql(
            """
            SELECT
              repo,
              commit,
              CAST(caller_goid_h128 AS BIGINT) AS caller_goid_h128,
              CAST(callee_goid_h128 AS BIGINT) AS callee_goid_h128,
              * EXCLUDE (repo, commit, caller_goid_h128, callee_goid_h128)
            FROM graph.call_graph_edges
            """
        )
    if table_name == "graph.symbol_use_edges":
        repo_default, commit_default = _default_repo_commit(con)
        df = con.execute(
            """
            SELECT
              ? AS repo,
              ? AS commit,
              symbol,
              def_path,
              use_path,
              same_file,
              same_module,
              CAST(def_goid_h128 AS BIGINT) AS def_goid_h128,
              CAST(use_goid_h128 AS BIGINT) AS use_goid_h128
            FROM graph.symbol_use_edges
            """,
            [repo_default, commit_default],
        ).fetch_df()
        return con.from_df(df)
    if table_name == "analytics.test_coverage_edges":
        return con.sql(
            """
            SELECT
              repo,
              commit,
              test_id,
              CAST(test_goid_h128 AS BIGINT) AS test_goid_h128,
              CAST(function_goid_h128 AS BIGINT) AS function_goid_h128,
              urn,
              rel_path,
              qualname,
              covered_lines,
              executable_lines,
              coverage_ratio,
              last_status,
              CAST(created_at AS VARCHAR) AS created_at
            FROM analytics.test_coverage_edges
            """
        )
    return con.table(table_name)


def export_jsonl_for_table(
    gateway: StorageGateway,
    table_name: str,
    output_path: Path,
) -> None:
    """
    Export a single DuckDB table to JSONL.

    Parameters
    ----------
    gateway :
        StorageGateway providing the DuckDB connection.
    table_name : str
        Fully qualified table name (schema.table) to export.
    output_path : Path
        Destination path for the JSONL file.

    Notes
    -----
    Uses `COPY (SELECT * FROM <table>) TO <path> (FORMAT JSON, ARRAY FALSE)`
    so each row is serialized as a single JSON object per line. Table names are
    validated against the known dataset mapping to avoid unsafe SQL injection.

    Raises
    ------
    ValueError
        If the requested table is not in the allowed export mapping.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if table_name not in JSONL_DATASETS:
        message = f"Refusing to export unknown table: {table_name}"
        raise ValueError(message)
    log.info("Exporting %s -> %s", table_name, output_path)
    con = gateway.con

    if table_name == "analytics.function_validation":
        row = con.execute("SELECT COUNT(*) FROM analytics.function_validation").fetchone()
        count = int(row[0]) if row is not None else 0
        if count == 0:
            payload = {
                "message": "No function validation issues found.",
                "error_types": ["parse_failed", "span_not_found"],
                "generated_at": datetime.now(UTC).isoformat(),
            }
            output_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            return

    rel = _normalized_relation(con, table_name)
    write_json = getattr(rel, "write_json", None)
    if write_json is not None:
        callable_write_json = cast("Callable[..., object]", write_json)
        callable_write_json(str(output_path), array=False)
        return

    df = rel.df()
    df.to_json(output_path, orient="records", lines=True, date_format="iso")


def export_all_jsonl(
    gateway: StorageGateway,
    document_output_dir: Path,
    *,
    validate_exports: bool = True,
    schemas: list[str] | None = None,
) -> list[Path]:
    """
    Export all configured datasets to JSONL files under `Document Output/`.

    Parameters
    ----------
    gateway :
        StorageGateway providing the DuckDB connection seeded with CodeIntel schemas.
    document_output_dir : Path
        Target directory where JSONL artifacts are written.
    validate_exports : bool
        Whether to validate selected datasets against JSON Schemas after export.
    schemas : list[str] | None
        Optional subset of schema names to validate; defaults to the standard export set.

    Returns
    -------
    list[Path]
        Paths to every JSON/JSONL file written, including the manifest.

    Raises
    ------
    ExportError
        If validation fails for any selected schema.
    """
    document_output_dir = document_output_dir.resolve()
    document_output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    for table_name, filename in JSONL_DATASETS.items():
        output_path = document_output_dir / filename
        try:
            export_jsonl_for_table(gateway, table_name, output_path)
            written.append(output_path)
        except (duckdb.Error, OSError, ValueError) as exc:
            # Log and continue: some tables may legitimately be empty or missing
            log.warning(
                "Failed to export %s to %s: %s",
                table_name,
                output_path,
                exc,
            )

    # repo_map.json is handled separately
    repo_map_path = export_repo_map_json(gateway, document_output_dir)
    if repo_map_path is not None:
        written.append(repo_map_path)

    # Optionally write a small manifest
    index_path = document_output_dir / "index.json"
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "files": [p.name for p in written],
    }
    index_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(index_path)

    if validate_exports:
        schema_list = schemas or DEFAULT_VALIDATION_SCHEMAS
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

    return written


def export_repo_map_json(
    gateway: StorageGateway,
    document_output_dir: Path,
) -> Path | None:
    """
    Export `core.repo_map` to a `repo_map.json` file.

    Extended Summary
    ----------------
    The payload mirrors the structure described in README_METADATA, including
    repo identifiers, module mapping, overlays, and a generation timestamp:

      {
        "repo": "...",
        "commit": "...",
        "modules": { "pkg.mod": "path/to/file.py", ... },
        "overlays": {...},
        "generated_at": "2024-01-01T00:00:00Z"
      } :contentReference[oaicite:6]{index=6}

    Returns
    -------
    Path | None
        Path to the written `repo_map.json`, or None when no repo_map rows
        are available in the database.
    """
    document_output_dir = document_output_dir.resolve()
    document_output_dir.mkdir(parents=True, exist_ok=True)

    con = gateway.con
    df = con.execute(
        "SELECT repo, commit, modules, overlays, generated_at FROM core.repo_map"
    ).fetch_df()
    if df.empty:
        log.warning("core.repo_map is empty; skipping repo_map.json export")
        return None

    # For now we export the first row; typical usage is one repo/commit per DB.
    row = df.iloc[0]
    payload = {
        "repo": row["repo"],
        "commit": row["commit"],
        "modules": row["modules"],
        "overlays": row.get("overlays") if "overlays" in df.columns else {},
        "generated_at": row["generated_at"].isoformat()
        if hasattr(row["generated_at"], "isoformat")
        else str(row["generated_at"]),
    }

    output_path = document_output_dir / "repo_map.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Exported repo_map.json -> %s", output_path)
    return output_path
