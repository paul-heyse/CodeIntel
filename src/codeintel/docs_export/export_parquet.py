"""Parquet exporters for the CodeIntel metadata warehouse."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

from codeintel.docs_export import DEFAULT_VALIDATION_SCHEMAS
from codeintel.docs_export.datasets import JSONL_DATASETS, PARQUET_DATASETS
from codeintel.docs_export.manifest import write_dataset_manifest
from codeintel.docs_export.validate_exports import validate_files
from codeintel.server.datasets import validate_dataset_registry
from codeintel.services.errors import ExportError, problem
from codeintel.storage.gateway import (
    DuckDBConnection,
    DuckDBError,
    DuckDBRelation,
    StorageGateway,
)

log = logging.getLogger(__name__)


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
        return {name: table for name, table in dataset_mapping.items() if table in PARQUET_DATASETS}
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


def export_parquet_for_table(
    gateway: StorageGateway,
    table_name: str,
    output_path: Path,
) -> None:
    """
    Export a single DuckDB table to Parquet.

    Parameters
    ----------
    gateway :
        StorageGateway providing the DuckDB connection.
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
        If the requested table is not registered in the dataset mapping.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_mapping = gateway.datasets.mapping
    if table_name not in dataset_mapping.values():
        message = f"Refusing to export unknown dataset table: {table_name}"
        raise ValueError(message)
    log.info("Exporting %s -> %s", table_name, output_path)
    rel = _normalized_relation(gateway.con, table_name)
    rel.write_parquet(str(output_path))


def export_dataset_to_parquet(
    gateway: StorageGateway,
    dataset_name: str,
    output_dir: Path,
) -> Path:
    """
    Export a dataset resolved through the dataset registry to Parquet.

    Parameters
    ----------
    gateway :
        StorageGateway providing the DuckDB connection.
    dataset_name : str
        Logical dataset name to export (e.g., ``function_profile``).
    output_dir : Path
        Destination directory for the Parquet file.

    Returns
    -------
    Path
        Path to the written Parquet file.

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
    filename = PARQUET_DATASETS.get(table_name, f"{dataset_name}.parquet")
    output_path = output_dir / filename
    export_parquet_for_table(gateway, table_name, output_path)
    return output_path


def export_all_parquet(
    gateway: StorageGateway,
    document_output_dir: Path,
    *,
    validate_exports: bool = True,
    schemas: list[str] | None = None,
    datasets: list[str] | None = None,
) -> None:
    """
    Export configured datasets to Parquet files under `Document Output/`.

    Parameters
    ----------
    gateway:
        StorageGateway with all tables populated.
    document_output_dir:
        Path to the `Document Output/` directory (will be created
        if it does not exist).
    validate_exports:
        Whether to validate selected datasets against JSON Schemas after export.
    schemas:
        Optional subset of schema names to validate; defaults to the standard export set.
    datasets:
        Optional list of dataset names to export; defaults to datasets with Parquet filenames.

    Raises
    ------
    ExportError
        If validation fails for any selected schema.
    """
    document_output_dir = document_output_dir.resolve()
    document_output_dir.mkdir(parents=True, exist_ok=True)

    validate_dataset_registry(gateway)
    dataset_mapping = gateway.datasets.mapping
    selected = _select_dataset_tables(dataset_mapping, datasets)
    missing_tables = set(PARQUET_DATASETS) - set(dataset_mapping.values())
    for table_name in sorted(missing_tables):
        log.warning("Skipping %s; table not present in dataset registry", table_name)

    written: list[Path] = []

    for dataset_name, table_name in sorted(selected.items()):
        filename = PARQUET_DATASETS.get(table_name, f"{dataset_name}.parquet")
        output_path = document_output_dir / filename
        try:
            export_parquet_for_table(gateway, table_name, output_path)
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
                raise ExportError(pd)
