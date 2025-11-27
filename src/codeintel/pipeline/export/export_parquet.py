"""Parquet exporters for the CodeIntel metadata warehouse."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from codeintel.pipeline.export import DEFAULT_VALIDATION_SCHEMAS
from codeintel.pipeline.export.export_jsonl import ExportCallOptions
from codeintel.pipeline.export.manifest import write_dataset_manifest
from codeintel.pipeline.export.validate_exports import validate_files
from codeintel.serving.http.datasets import validate_dataset_registry
from codeintel.serving.services.errors import ExportError, log_problem, problem
from codeintel.storage.gateway import (
    DuckDBConnection,
    DuckDBError,
    DuckDBRelation,
    StorageGateway,
)

log = logging.getLogger(__name__)

MAX_EXPORT_LIMIT = 9_223_372_036_854_775_807
AUDIT_LOG_PATH = os.getenv("CODEINTEL_EXPORT_AUDIT_LOG")
AUDIT_TABLE_ENABLED = os.getenv("CODEINTEL_EXPORT_AUDIT_TABLE") is not None
# When set, the above enable audit logging of export metadata.


@dataclass(frozen=True)
class AuditRecord:
    """Metadata about a Parquet export for optional audit logging."""

    table_name: str
    macro: str
    rows: int | None
    duration_s: float
    output_path: Path


def _validate_registry_or_raise(gateway: StorageGateway) -> None:
    """
    Validate dataset registry and normalize error type for schema mismatches.

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
    parquet_mapping: Mapping[str, str],
    datasets: list[str] | None,
) -> dict[str, str]:
    if datasets is None:
        return {name: table for name, table in dataset_mapping.items() if table in parquet_mapping}
    selected: dict[str, str] = {}
    for dataset_name in datasets:
        selected[dataset_name] = _resolve_dataset_table(dataset_name, dataset_mapping)
    return selected


NORMALIZED_MACROS: dict[str, str] = {
    "graph.call_graph_edges": "metadata.normalized_call_graph_edges",
    "graph.symbol_use_edges": "metadata.normalized_symbol_use_edges",
    "analytics.test_coverage_edges": "metadata.normalized_test_coverage_edges",
    "analytics.function_metrics": "metadata.normalized_function_metrics",
    "analytics.function_profile": "metadata.normalized_function_profile",
    "analytics.function_history": "metadata.normalized_function_history",
    "analytics.function_types": "metadata.normalized_function_types",
    "analytics.test_catalog": "metadata.normalized_test_catalog",
    "analytics.coverage_functions": "metadata.normalized_coverage_functions",
    "analytics.goid_risk_factors": "metadata.normalized_goid_risk_factors",
    "analytics.function_validation": "metadata.normalized_function_validation",
    "analytics.graph_validation": "metadata.normalized_graph_validation",
    "graph.cfg_blocks": "metadata.normalized_cfg_blocks",
    "graph.cfg_edges": "metadata.normalized_cfg_edges",
    "graph.dfg_edges": "metadata.normalized_dfg_edges",
    "analytics.cfg_block_metrics": "metadata.normalized_cfg_block_metrics",
    "analytics.cfg_function_metrics": "metadata.normalized_cfg_function_metrics",
    "analytics.cfg_function_metrics_ext": "metadata.normalized_cfg_function_metrics_ext",
    "analytics.config_data_flow": "metadata.normalized_config_data_flow",
    "analytics.data_model_usage": "metadata.normalized_data_model_usage",
    "analytics.dfg_block_metrics": "metadata.normalized_dfg_block_metrics",
    "analytics.dfg_function_metrics": "metadata.normalized_dfg_function_metrics",
    "analytics.dfg_function_metrics_ext": "metadata.normalized_dfg_function_metrics_ext",
    "analytics.entrypoint_tests": "metadata.normalized_entrypoint_tests",
    "analytics.entrypoints": "metadata.normalized_entrypoints",
    "analytics.external_dependency_calls": "metadata.normalized_external_dependency_calls",
    "analytics.function_contracts": "metadata.normalized_function_contracts",
    "analytics.function_effects": "metadata.normalized_function_effects",
    "analytics.graph_metrics_functions": "metadata.normalized_graph_metrics_functions",
    "analytics.graph_metrics_functions_ext": "metadata.normalized_graph_metrics_functions_ext",
    "analytics.history_timeseries": "metadata.normalized_history_timeseries",
    "analytics.semantic_roles_functions": "metadata.normalized_semantic_roles_functions",
    "analytics.symbol_graph_metrics_functions": "metadata.normalized_symbol_graph_metrics_functions",
    "analytics.test_graph_metrics_functions": "metadata.normalized_test_graph_metrics_functions",
    "analytics.test_profile": "metadata.normalized_test_profile",
    "analytics.behavioral_coverage": "metadata.normalized_behavioral_coverage",
}


def _macro_relation(
    con: DuckDBConnection,
    table_key: str,
    row_limit: int,
    row_offset: int,
    *,
    require_normalized_macros: bool,
) -> tuple[DuckDBRelation, str]:
    macro = NORMALIZED_MACROS.get(table_key)
    if macro:
        log.debug("Exporting via normalized macro for %s: %s", table_key, macro)
        try:
            return (
                con.sql(
                    f"SELECT * FROM {macro}(?, ?, ?)",  # noqa: S608 - trusted macro name
                    params=[table_key, row_limit, row_offset],
                ),
                macro,
            )
        except DuckDBError as exc:
            if require_normalized_macros:
                message = f"No normalized macro available for {table_key}"
                raise ValueError(message) from exc
            log.warning(
                "Normalized macro %s missing; falling back to dataset_rows for %s",
                macro,
                table_key,
            )
    if require_normalized_macros:
        message = f"No normalized macro found for {table_key}"
        raise ValueError(message)
    log.debug("Exporting via dataset_rows fallback for %s", table_key)
    return (
        con.sql(
            """
            SELECT *
            FROM metadata.dataset_rows(?, ?, ?)
            """,
            params=[table_key, row_limit, row_offset],
        ),
        "metadata.dataset_rows",
    )


def _write_audit_entry(
    *,
    record: AuditRecord,
    con: DuckDBConnection,
) -> None:
    if AUDIT_LOG_PATH is None and not AUDIT_TABLE_ENABLED:
        return
    json_record = {
        "table": record.table_name,
        "macro": record.macro,
        "rows": record.rows,
        "duration_s": record.duration_s,
        "output": str(record.output_path),
    }
    if AUDIT_LOG_PATH is not None:
        with Path(AUDIT_LOG_PATH).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_record))
            handle.write("\n")
    if AUDIT_TABLE_ENABLED:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata.export_audit (
                dataset TEXT,
                macro TEXT,
                rows BIGINT,
                duration_s DOUBLE,
                output_path TEXT,
                sql TEXT,
                plan TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        con.execute(
            """
            INSERT INTO metadata.export_audit
                (dataset, macro, rows, duration_s, output_path, sql, plan)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.table_name,
                record.macro,
                record.rows,
                record.duration_s,
                str(record.output_path),
                None,
                None,
            ],
        )


def export_parquet_for_table(
    gateway: StorageGateway,
    table_name: str,
    output_path: Path,
    *,
    require_normalized_macros: bool = False,
) -> None:
    """
    Export a single DuckDB table to Parquet.

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
    start = perf_counter()
    rel, macro_name = _macro_relation(
        gateway.con,
        table_name,
        MAX_EXPORT_LIMIT,
        0,
        require_normalized_macros=require_normalized_macros,
    )
    write_parquet = getattr(rel, "write_parquet", None)
    if write_parquet is not None:
        write_parquet(str(output_path))
        row_count_row = rel.aggregate("count(*)").fetchone()
        rows = int(row_count_row[0]) if row_count_row else 0
        duration = perf_counter() - start
        _write_audit_entry(
            record=AuditRecord(
                table_name=table_name,
                macro=macro_name,
                rows=rows,
                duration_s=duration,
                output_path=output_path,
            ),
            con=gateway.con,
        )
        log.debug(
            "Exported %s rows for %s via macro in %.3fs",
            rows,
            table_name,
            duration,
        )
        return
    df = rel.df()
    df.to_parquet(output_path)
    duration = perf_counter() - start
    rows = len(df)
    _write_audit_entry(
        record=AuditRecord(
            table_name=table_name,
            macro=macro_name,
            rows=rows,
            duration_s=duration,
            output_path=output_path,
        ),
        con=gateway.con,
    )
    log.debug("Exported %s rows for %s via macro fallback in %.3fs", rows, table_name, duration)


def export_dataset_to_parquet(
    gateway: StorageGateway,
    dataset_name: str,
    output_dir: Path,
    *,
    options: ExportCallOptions | None = None,
) -> Path:
    """
    Export a dataset resolved through the dataset registry to Parquet.

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
    parquet_mapping = gateway.datasets.parquet_mapping or {}
    if dataset_name not in dataset_mapping:
        message = f"Unknown dataset: {dataset_name}"
        raise ValueError(message)
    table_name = dataset_mapping[dataset_name]
    filename = parquet_mapping.get(table_name, f"{dataset_name}.parquet")
    output_path = output_dir / filename
    opts = options or ExportCallOptions()
    export_parquet_for_table(
        gateway,
        table_name,
        output_path,
        require_normalized_macros=opts.require_normalized_macros,
    )
    return output_path


def export_all_parquet(
    gateway: StorageGateway,
    document_output_dir: Path,
    *,
    options: ExportCallOptions | None = None,
) -> None:
    """
    Export configured datasets to Parquet files under `Document Output/`.

    Raises
    ------
    ExportError
        If validation fails for any selected schema after export.
    """
    opts = options or ExportCallOptions()
    document_output_dir = document_output_dir.resolve()
    document_output_dir.mkdir(parents=True, exist_ok=True)

    _validate_registry_or_raise(gateway)
    dataset_mapping = gateway.datasets.mapping
    jsonl_mapping = gateway.datasets.jsonl_mapping or {}
    parquet_mapping = gateway.datasets.parquet_mapping or {}
    selected = _select_dataset_tables(dataset_mapping, parquet_mapping, opts.datasets)
    missing_tables = set(parquet_mapping) - set(dataset_mapping.values())
    for table_name in sorted(missing_tables):
        log.warning("Skipping %s; table not present in dataset registry", table_name)

    written: list[Path] = []

    for dataset_name, table_name in sorted(selected.items()):
        filename = parquet_mapping.get(table_name, f"{dataset_name}.parquet")
        output_path = document_output_dir / filename
        try:
            export_parquet_for_table(
                gateway,
                table_name,
                output_path,
                require_normalized_macros=opts.require_normalized_macros,
            )
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
        jsonl_mapping=jsonl_mapping,
        parquet_mapping=parquet_mapping,
        selected=list(selected.keys()),
    )
    written.append(manifest_path)

    if opts.validate_exports:
        schema_list = opts.schemas or DEFAULT_VALIDATION_SCHEMAS
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
