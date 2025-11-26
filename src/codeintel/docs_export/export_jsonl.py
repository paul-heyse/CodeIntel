"""JSON/JSONL exporters for the CodeIntel metadata warehouse."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from codeintel.docs_export import DEFAULT_VALIDATION_SCHEMAS
from codeintel.docs_export.datasets import JSONL_DATASETS, PARQUET_DATASETS
from codeintel.docs_export.manifest import write_dataset_manifest
from codeintel.docs_export.validate_exports import validate_files
from codeintel.server.datasets import validate_dataset_registry
from codeintel.services.errors import ExportError, log_problem, problem
from codeintel.storage.gateway import (
    DuckDBConnection,
    DuckDBError,
    DuckDBRelation,
    StorageGateway,
)

log = logging.getLogger(__name__)


def _validate_registry_or_raise(gateway: StorageGateway) -> None:
    """Validate dataset registry and normalize error type for schema mismatches.

    Raises
    ------
    ValueError
        If required tables or views are missing from the registry.
    ExportError
        If tables exist but their schemas do not match expectations.
    """
    try:
        validate_dataset_registry(gateway)
    except ValueError as exc:
        detail = str(exc)
        pd = problem(
            code="export.validation_failed",
            title="Export validation failed",
            detail=detail,
            extras={"stage": "dataset_registry"},
        )
        log_problem(log, pd)
        if "schema mismatches" in detail:
            raise ExportError(pd) from exc
        raise


def _resolve_dataset_table(dataset_name: str, dataset_mapping: Mapping[str, str]) -> str:
    table = dataset_mapping.get(dataset_name)
    if table is None:
        message = f"Unknown dataset: {dataset_name}"
        raise ValueError(message)
    return table


def _select_dataset_tables(
    dataset_mapping: Mapping[str, str],
    datasets: list[str] | None,
) -> dict[str, str]:
    if datasets is None:
        return {name: table for name, table in dataset_mapping.items() if table in JSONL_DATASETS}
    selected: dict[str, str] = {}
    for dataset_name in datasets:
        selected[dataset_name] = _resolve_dataset_table(dataset_name, dataset_mapping)
    return selected


def _default_repo_commit(con: DuckDBConnection) -> tuple[str, str]:
    row = con.execute("SELECT repo, commit FROM core.repo_map LIMIT 1").fetchone()
    if row is None:
        return "", ""
    repo, commit = row
    return str(repo), str(commit)


def _normalized_relation(con: DuckDBConnection, table_name: str) -> DuckDBRelation:
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
        If the requested table is not registered in the dataset mapping.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_mapping = gateway.datasets.mapping
    if table_name not in dataset_mapping.values():
        message = f"Refusing to export unknown dataset table: {table_name}"
        raise ValueError(message)
    log.info("Exporting %s -> %s", table_name, output_path)
    con = gateway.con

    if table_name == "analytics.function_validation":
        row = con.execute("SELECT COUNT(*) FROM analytics.function_validation").fetchone()
        count = int(row[0]) if row is not None else 0
        if count == 0:
            payload = {
                "message": "No function validation issues found.",
                "error_types": ["parse_failed", "span_not_found", "unknown_function"],
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


def export_dataset_to_jsonl(
    gateway: StorageGateway,
    dataset_name: str,
    output_dir: Path,
) -> Path:
    """
    Export a dataset resolved through the dataset registry to JSONL.

    Parameters
    ----------
    gateway :
        StorageGateway providing the DuckDB connection.
    dataset_name : str
        Logical dataset name to export (e.g., ``function_profile``).
    output_dir : Path
        Destination directory for the JSONL file.

    Returns
    -------
    Path
        Path to the written JSONL file.

    Raises
    ------
    ValueError
        If the dataset name is unknown.
    """
    dataset_mapping = gateway.datasets.mapping
    if dataset_name not in dataset_mapping:
        message = f"Unknown dataset: {dataset_name}"
        raise ValueError(message)
    table_name = dataset_mapping[dataset_name]
    filename = JSONL_DATASETS.get(table_name, f"{dataset_name}.jsonl")
    output_path = output_dir / filename
    export_jsonl_for_table(gateway, table_name, output_path)
    return output_path


def export_all_jsonl(
    gateway: StorageGateway,
    document_output_dir: Path,
    *,
    validate_exports: bool = True,
    schemas: list[str] | None = None,
    datasets: list[str] | None = None,
) -> list[Path]:
    """
    Export configured datasets to JSONL files under `Document Output/`.

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
    datasets : list[str] | None
        Optional list of dataset names to export; defaults to datasets with JSONL filenames.

    Returns
    -------
    list[Path]
        Paths to every JSON/JSONL file written, including the manifest.

    Raises
    ------
    ExportError
        If validation fails for any selected schema after export.
    """
    document_output_dir = document_output_dir.resolve()
    document_output_dir.mkdir(parents=True, exist_ok=True)

    _validate_registry_or_raise(gateway)
    dataset_mapping = gateway.datasets.mapping
    selected = _select_dataset_tables(dataset_mapping, datasets)
    missing_tables = set(JSONL_DATASETS) - set(dataset_mapping.values())
    for table_name in sorted(missing_tables):
        log.warning("Skipping %s; table not present in dataset registry", table_name)

    written: list[Path] = []

    for dataset_name, table_name in sorted(selected.items()):
        filename = JSONL_DATASETS.get(table_name, f"{dataset_name}.jsonl")
        output_path = document_output_dir / filename
        try:
            export_jsonl_for_table(gateway, table_name, output_path)
            written.append(output_path)
        except (DuckDBError, OSError, ValueError) as exc:
            # Log and continue: some tables may legitimately be empty or missing
            log.warning(
                "Failed to export dataset %s (%s) to %s: %s",
                dataset_name,
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
    manifest_path = write_dataset_manifest(
        document_output_dir,
        dataset_mapping,
        jsonl_mapping=JSONL_DATASETS,
        parquet_mapping=PARQUET_DATASETS,
        selected=list(selected.keys()),
    )
    written.append(manifest_path)

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
                log_problem(log, pd)
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
