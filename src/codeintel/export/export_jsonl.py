"""JSON/JSONL exporters for the CodeIntel metadata warehouse."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal, cast

from codeintel.export import default_validation_schemas
from codeintel.export.export_exprs import build_export_expr, compile_export_sql
from codeintel.export.manifest import (
    ExportManifestData,
    IncrementalMarker,
    SkipCriteria,
    compute_file_hash,
    read_incremental_marker,
    should_skip_export,
    write_dataset_manifest,
    write_incremental_marker,
    write_per_dataset_manifest,
)
from codeintel.export.validate_exports import validate_files
from codeintel.serving.backend.datasets import validate_dataset_registry
from codeintel.serving.services.errors import ExportError, ProblemDetails, log_problem, problem
from codeintel.storage.gateway import (
    DuckDBError,
)
from codeintel.storage.validation import _schema_path

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from codeintel.config.datasets import DatasetContract
    from codeintel.storage.gateway import (
        DuckDBConnection,
        DuckDBRelation,
        StorageGateway,
    )

log = logging.getLogger(__name__)

MAX_EXPORT_LIMIT = 9_223_372_036_854_775_807
AUDIT_LOG_PATH = os.getenv("CODEINTEL_EXPORT_AUDIT_LOG")
AUDIT_TABLE_ENABLED = os.getenv("CODEINTEL_EXPORT_AUDIT_TABLE") is not None


@dataclass(frozen=True)
class AuditRecord:
    """Metadata about a completed export for optional audit logging."""

    table_name: str
    macro: str
    rows: int | None
    duration_s: float
    output_path: Path


@dataclass(frozen=True)
class ExportCallOptions:
    """Options controlling dataset selection, validation, and macro enforcement."""

    validate_exports: bool = True
    schemas: list[str] | None = None
    datasets: list[str] | None = None
    validation_profile: Literal["strict", "lenient"] | None = None
    force_full_export: bool = False


@dataclass(frozen=True)
class ExportTarget:
    """Inputs describing a dataset export request."""

    dataset_name: str
    table_name: str
    output_path: Path
    dataset: DatasetContract | None


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
            ProblemDetails(
                code="export.validation_failed",
                title="Export validation failed",
                detail=detail,
                extras={"stage": "dataset_registry"},
            )
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
    jsonl_mapping: Mapping[str, str],
    datasets: list[str] | None,
) -> dict[str, str]:
    """
    Determine which dataset names and tables to export.

    Parameters
    ----------
    dataset_mapping
        Mapping of dataset name -> table/view key from the gateway registry.
    jsonl_mapping
        Mapping of table/view key -> JSONL filename from the gateway registry.
    datasets
        Optional list of dataset names requested by the caller.

    Returns
    -------
    dict[str, str]
        Selected dataset name -> table/view key mapping.
    """
    if datasets is None:
        return {name: table for name, table in dataset_mapping.items() if table in jsonl_mapping}
    selected: dict[str, str] = {}
    for dataset_name in datasets:
        selected[dataset_name] = _resolve_dataset_table(dataset_name, dataset_mapping)
    return selected


def _resolve_validation_profile(
    options: ExportCallOptions,
    dataset: DatasetContract | None,
) -> str:
    if options.validation_profile is not None:
        return options.validation_profile
    if dataset is not None:
        return dataset.validation_profile
    return "strict"


def _schema_digest(dataset: DatasetContract | None) -> str | None:
    if dataset is None or dataset.json_schema_id is None:
        return None
    schema_file = _schema_path(dataset.json_schema_id)
    if not schema_file.exists():
        return None
    return compute_file_hash(schema_file)


def _row_count(gateway: StorageGateway, table_name: str) -> int | None:
    try:
        table = gateway.ibis.table(table_name)
        row = table.count().execute()
    except DuckDBError:
        log.debug("Row count unavailable for %s", table_name, exc_info=True)
        return None
    if row is None:
        return None
    return int(row[0])


def _export_relation(
    gateway: StorageGateway,
    table_key: str,
    row_limit: int,
    row_offset: int,
) -> DuckDBRelation:
    expr = build_export_expr(gateway, table_key, limit=row_limit, offset=row_offset)
    sql = compile_export_sql(expr)
    return gateway.con.sql(sql)


def _write_audit_entry(
    record: AuditRecord,
    *,
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

    start = perf_counter()
    rel = _export_relation(gateway, table_name, MAX_EXPORT_LIMIT, 0)
    macro_name = "ibis_export"
    write_json = getattr(rel, "write_json", None)
    if write_json is not None:
        callable_write_json = cast("Callable[..., object]", write_json)
        callable_write_json(str(output_path), array=False)
        row_count_row = rel.aggregate("count(*)").fetchone()
        rows = int(row_count_row[0]) if row_count_row else 0
        duration = perf_counter() - start
        _write_audit_entry(
            AuditRecord(
                table_name=table_name,
                macro=macro_name,
                rows=rows,
                duration_s=duration,
                output_path=output_path,
            ),
            con=con,
        )
        log.debug(
            "Exported %s rows for %s via Ibis export in %.3fs",
            rows,
            table_name,
            duration,
        )
        return

    df = rel.df()
    df.to_json(output_path, orient="records", lines=True, date_format="iso")
    duration = perf_counter() - start
    rows = len(df)
    _write_audit_entry(
        AuditRecord(
            table_name=table_name,
            macro=macro_name,
            rows=rows,
            duration_s=duration,
            output_path=output_path,
        ),
        con=con,
    )
    log.debug(
        "Exported %s rows for %s via Ibis export fallback in %.3fs", rows, table_name, duration
    )


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
    jsonl_mapping = gateway.datasets.jsonl_mapping or {}
    if dataset_name not in dataset_mapping:
        message = f"Unknown dataset: {dataset_name}"
        raise ValueError(message)
    table_name = dataset_mapping[dataset_name]
    filename = jsonl_mapping.get(table_name, f"{dataset_name}.jsonl")
    output_path = output_dir / filename
    export_jsonl_for_table(
        gateway,
        table_name,
        output_path,
    )
    return output_path


def _export_dataset_jsonl(
    gateway: StorageGateway,
    target: ExportTarget,
    *,
    opts: ExportCallOptions,
) -> Path | None:
    if target.dataset is not None:
        caps = target.dataset.capabilities()
        if not caps["can_export_jsonl"]:
            log.warning("Skipping dataset %s; JSONL export not supported", target.dataset_name)
            return None
    validation_profile = _resolve_validation_profile(opts, target.dataset)
    schema_digest = _schema_digest(target.dataset)
    marker = read_incremental_marker(target.output_path)
    current_row_count: int | None = None
    if target.dataset is None or not target.dataset.is_view:
        current_row_count = _row_count(gateway, target.table_name)
    criteria = SkipCriteria(
        row_count=current_row_count,
        schema_version=target.dataset.schema_version if target.dataset else None,
        validation_profile=validation_profile,
        schema_digest=schema_digest,
        force_full_export=opts.force_full_export,
    )
    if should_skip_export(marker, criteria):
        log.info(
            "Skipping dataset %s export; marker matches row_count=%s, schema_version=%s",
            target.dataset_name,
            current_row_count,
            marker.get("schema_version") if marker else None,
        )
        if target.output_path.exists():
            return target.output_path
        return None
    try:
        started_at = datetime.now(UTC)
        export_jsonl_for_table(
            gateway,
            target.table_name,
            target.output_path,
        )
        data_hash = compute_file_hash(target.output_path)
        completed_at = datetime.now(UTC)
        final_row_count = (
            current_row_count
            if current_row_count is not None
            else _row_count(gateway, target.table_name)
        )
    except (DuckDBError, OSError, ValueError) as exc:
        log.warning(
            "Failed to export dataset %s (%s) to %s: %s",
            target.dataset_name,
            target.table_name,
            target.output_path,
            exc,
        )
        return None
    manifest_payload = ExportManifestData(
        dataset=target.dataset_name,
        schema_id=target.dataset.json_schema_id if target.dataset else None,
        schema_version=target.dataset.schema_version if target.dataset else None,
        schema_digest=schema_digest,
        validation_profile=validation_profile,
        row_count=final_row_count or 0,
        data_hash=data_hash,
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
    )
    write_per_dataset_manifest(target.output_path, manifest_payload)
    write_incremental_marker(
        target.output_path,
        IncrementalMarker(
            dataset=target.dataset_name,
            row_count=final_row_count or 0,
            schema_version=target.dataset.schema_version if target.dataset else None,
            validation_profile=validation_profile,
            schema_digest=schema_digest,
        ),
    )
    return target.output_path


def _validate_written_exports(
    written: list[Path],
    registry_meta: Mapping[str, DatasetContract],
    opts: ExportCallOptions,
) -> None:
    if not opts.validate_exports:
        return
    schema_names = opts.schemas or default_validation_schemas()
    for schema_name in schema_names:
        matching = [p for p in written if p.name.startswith(schema_name)]
        if not matching:
            continue
        ds = registry_meta.get(schema_name)
        if ds is None or ds.json_schema_id is None:
            log.info("Skipping validation for %s; no JSON Schema configured", schema_name)
            continue
        profile = _resolve_validation_profile(opts, ds)
        result = validate_files(schema_name, matching)
        if result != 0 and profile == "lenient":
            pd = problem(
                ProblemDetails(
                    code="export.validation_failed",
                    title="Export validation failed",
                    detail=f"Validation failed for schema {schema_name}",
                    extras={"schema": schema_name, "files": [str(p) for p in matching]},
                )
            )
            log_problem(log, pd)
            continue
        if result != 0:
            pd = problem(
                ProblemDetails(
                    code="export.validation_failed",
                    title="Export validation failed",
                    detail=f"Validation failed for schema {schema_name}",
                    extras={"schema": schema_name, "files": [str(p) for p in matching]},
                )
            )
            log_problem(log, pd)
            raise ExportError(pd)


def export_all_jsonl(
    gateway: StorageGateway,
    document_output_dir: Path,
    *,
    options: ExportCallOptions | None = None,
) -> list[Path]:
    """
    Export configured datasets to JSONL files under `Document Output/`.

    Parameters
    ----------
    gateway :
        StorageGateway providing the DuckDB connection seeded with CodeIntel schemas.
    document_output_dir : Path
        Target directory where JSONL artifacts are written.
    options : ExportCallOptions | None
        Export options controlling dataset selection, validation, and macro requirements.

    Returns
    -------
    list[Path]
        Paths to every JSON/JSONL file written, including the manifest.
    """
    document_output_dir = document_output_dir.resolve()
    document_output_dir.mkdir(parents=True, exist_ok=True)

    opts = options or ExportCallOptions()
    _validate_registry_or_raise(gateway)
    dataset_mapping = gateway.datasets.mapping
    jsonl_mapping = gateway.datasets.jsonl_mapping or {}
    registry_meta = gateway.datasets.meta or {}
    selected = _select_dataset_tables(dataset_mapping, jsonl_mapping, opts.datasets)
    for table_name in sorted(set(jsonl_mapping) - set(dataset_mapping.values())):
        log.warning("Skipping %s; table not present in dataset registry", table_name)

    written: list[Path] = []

    for dataset_name, table_name in sorted(selected.items()):
        target = ExportTarget(
            dataset_name=dataset_name,
            table_name=table_name,
            output_path=document_output_dir
            / jsonl_mapping.get(table_name, f"{dataset_name}.jsonl"),
            dataset=registry_meta.get(dataset_name),
        )
        exported = _export_dataset_jsonl(
            gateway,
            target,
            opts=opts,
        )
        if exported is not None:
            written.append(exported)

    if repo_map := export_repo_map_json(gateway, document_output_dir):
        written.append(repo_map)

    index_path = document_output_dir / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "files": [p.name for p in written],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    written.append(index_path)
    manifest_path = write_dataset_manifest(
        document_output_dir,
        dataset_mapping,
        jsonl_mapping=jsonl_mapping,
        parquet_mapping=gateway.datasets.parquet_mapping or {},
        selected=list(selected.keys()),
    )
    written.append(manifest_path)

    _validate_written_exports(written, registry_meta, opts)

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
