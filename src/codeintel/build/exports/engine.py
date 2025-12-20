"""Export engine for JSONL/Parquet dataset exports.

This module centralizes the shared export control flow:

- dataset selection from registry
- incremental marker read/write
- per-dataset manifest writing
- optional schema validation
- audit logging

Format-specific serialization is delegated to writer callables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, Literal

from codeintel.build.errors import BuildProblemError
from codeintel.build.exports.common import (
    AUDIT_LOG_PATH,
    AUDIT_TABLE_ENABLED,
    MAX_EXPORT_LIMIT,
    AuditRecord,
    ExportCallOptions,
    ExportTarget,
    build_export_relation,
    compute_schema_digest,
    default_validation_schemas,
    log_export_error,
    resolve_validation_profile,
    select_dataset_tables,
    validate_registry_or_raise,
    write_audit_entry,
)
from codeintel.build.exports.manifest import (
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
from codeintel.build.exports.validation import validate_export_files
from codeintel.build.exports.writers import (
    default_json_serializer,
    write_jsonl_records,
    write_parquet_relation,
)
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.queries.safe import safe_count

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

ExportFormat = Literal["jsonl", "parquet"]

_EXPORT_RECORD_BATCH_SIZE = 10_000


@dataclass(frozen=True, slots=True)
class _ExportFormatSpec:
    format: ExportFormat
    mapping: Mapping[str, str]
    can_export_capability_key: str
    extension: str
    write_table: Callable[[StorageGateway, str, Path], int]


def export_jsonl_for_table(
    gateway: StorageGateway,
    table_key: str,
    output_path: Path,
) -> int:
    """Export a DuckDB table to JSONL.

    Parameters
    ----------
    gateway
        Storage gateway providing the DuckDB connection.
    table_key
        Fully qualified table key to export (schema.table).
    output_path
        Output JSONL path.

    Returns
    -------
    int
        Number of rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    rel = build_export_relation(gateway, table_key, MAX_EXPORT_LIMIT, 0)
    rows_written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        rows_written = write_jsonl_records(
            handle,
            rel=rel,
            serializer=default_json_serializer,
            batch_size=_EXPORT_RECORD_BATCH_SIZE,
        )
    duration = perf_counter() - start
    write_audit_entry(
        AuditRecord(
            table_name=table_key,
            macro="duckdb_relation",
            rows=rows_written,
            duration_s=duration,
            output_path=output_path,
        ),
        con=gateway.con,
    )
    return rows_written


def export_parquet_for_table(
    gateway: StorageGateway,
    table_key: str,
    output_path: Path,
) -> int:
    """Export a DuckDB table to Parquet.

    Parameters
    ----------
    gateway
        Storage gateway providing the DuckDB connection.
    table_key
        Fully qualified table key to export (schema.table).
    output_path
        Output Parquet path.

    Returns
    -------
    int
        Number of rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    rel = build_export_relation(gateway, table_key, MAX_EXPORT_LIMIT, 0)
    rows_written = write_parquet_relation(
        rel=rel,
        output_path=output_path,
        batch_size=_EXPORT_RECORD_BATCH_SIZE,
    )
    duration = perf_counter() - start
    write_audit_entry(
        AuditRecord(
            table_name=table_key,
            macro="duckdb_relation",
            rows=rows_written,
            duration_s=duration,
            output_path=output_path,
        ),
        con=gateway.con,
    )
    return rows_written


def _format_spec(gateway: StorageGateway, fmt: ExportFormat) -> _ExportFormatSpec:
    if fmt == "jsonl":
        return _ExportFormatSpec(
            format="jsonl",
            mapping=gateway.datasets.jsonl_datasets,
            can_export_capability_key="can_export_jsonl",
            extension=".jsonl",
            write_table=export_jsonl_for_table,
        )
    return _ExportFormatSpec(
        format="parquet",
        mapping=gateway.datasets.parquet_datasets,
        can_export_capability_key="can_export_parquet",
        extension=".parquet",
        write_table=export_parquet_for_table,
    )


def _validate_written_exports(
    written: list[Path],
    registry_by_table_key: Mapping[str, DatasetContract],
    opts: ExportCallOptions,
) -> None:
    if not opts.validate_exports:
        return
    table_keys = opts.schemas or default_validation_schemas()
    for table_key in table_keys:
        dataset = registry_by_table_key.get(table_key)
        if dataset is None:
            log.info("Skipping validation for %s; table key not in registry", table_key)
            continue
        dataset_name = dataset.name
        matching = [p for p in written if p.name.startswith(dataset_name)]
        if not matching:
            continue
        if dataset.json_schema_id is None:
            log.info("Skipping validation for %s; no JSON Schema configured", table_key)
            continue
        profile = resolve_validation_profile(opts, dataset)
        exit_code = validate_export_files(table_key, matching, dataset_name=dataset_name)
        if exit_code != 0 and profile == "lenient":
            log_export_error(
                code="export.validation_failed",
                title="Export validation failed",
                detail=f"Validation failed for schema {table_key}",
                table_key=table_key,
                files=[str(p) for p in matching],
            )
            continue
        if exit_code != 0:
            msg = f"Validation failed for schema {table_key}"
            log_export_error(
                code="export.validation_failed",
                title="Export validation failed",
                detail=msg,
                table_key=table_key,
                files=[str(p) for p in matching],
            )
            raise BuildProblemError.from_detail(
                code="export.validation_failed",
                title="Export validation failed",
                detail=msg,
                table_key=table_key,
                files=[str(p) for p in matching],
            )


def _export_dataset(
    gateway: StorageGateway,
    target: ExportTarget,
    *,
    spec: _ExportFormatSpec,
    opts: ExportCallOptions,
) -> Path | None:
    if target.dataset is not None:
        caps = target.dataset.capabilities()
        if not caps.get(spec.can_export_capability_key, False):
            log.warning(
                "Skipping dataset %s; %s export not supported",
                target.dataset_name,
                spec.format,
            )
            return None

    validation_profile = resolve_validation_profile(opts, target.dataset)
    schema_digest = compute_schema_digest(target.dataset)
    marker = read_incremental_marker(target.output_path)

    current_row_count: int | None = None
    if target.dataset is None or not target.dataset.is_view:
        current_row_count = safe_count(gateway, target.table_name)

    criteria = SkipCriteria(
        row_count=current_row_count,
        schema_version=target.dataset.schema_version if target.dataset else None,
        validation_profile=validation_profile,
        schema_digest=schema_digest,
        force_full_export=opts.force_full_export,
    )
    if should_skip_export(marker, criteria):
        if target.output_path.exists():
            return target.output_path
        return None

    try:
        started_at = datetime.now(UTC)
        rows_written = spec.write_table(gateway, target.table_name, target.output_path)
        data_hash = compute_file_hash(target.output_path)
        completed_at = datetime.now(UTC)
        final_row_count = (
            current_row_count
            if current_row_count is not None
            else safe_count(gateway, target.table_name)
        )
    except (DuckDBError, OSError, ValueError, TypeError) as exc:
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
        artifact=target.output_path.name,
        schema_id=target.dataset.json_schema_id if target.dataset else None,
        schema_version=target.dataset.schema_version if target.dataset else None,
        schema_digest=schema_digest,
        validation_profile=validation_profile,
        row_count=(final_row_count or rows_written or 0),
        data_hash=data_hash,
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
    )
    write_per_dataset_manifest(target.output_path, manifest_payload)
    write_incremental_marker(
        target.output_path,
        IncrementalMarker(
            dataset=target.dataset_name,
            row_count=(final_row_count or rows_written or 0),
            schema_version=target.dataset.schema_version if target.dataset else None,
            validation_profile=validation_profile,
            schema_digest=schema_digest,
        ),
    )
    return target.output_path


def export_all_datasets(
    gateway: StorageGateway,
    document_output_dir: Path,
    *,
    fmt: ExportFormat,
    options: ExportCallOptions | None = None,
) -> list[Path]:
    """Export configured datasets to a given format under `Document Output/`.

    Parameters
    ----------
    gateway
        Storage gateway providing the DuckDB connection and dataset registry.
    document_output_dir
        Root directory under which dataset artifacts are written.
    fmt
        Export format ("jsonl" or "parquet").
    options
        Export selection and validation options.

    Returns
    -------
    list[Path]
        Paths to written dataset artifacts and the top-level manifest.
    """
    opts = options or ExportCallOptions()
    document_output_dir = document_output_dir.resolve()
    document_output_dir.mkdir(parents=True, exist_ok=True)

    validate_registry_or_raise(gateway)
    registry = gateway.datasets
    dataset_mapping = {name: contract.table_key for name, contract in registry.by_name.items()}
    spec = _format_spec(gateway, fmt)
    registry_meta = registry.by_name

    selected = select_dataset_tables(dataset_mapping, spec.mapping, opts.datasets)
    missing_tables = set(spec.mapping) - set(dataset_mapping.values())
    for table_name in sorted(missing_tables):
        log.warning("Skipping %s; table not present in dataset registry", table_name)

    written: list[Path] = []
    for dataset_name, table_name in sorted(selected.items()):
        filename = spec.mapping.get(table_name, f"{dataset_name}{spec.extension}")
        target = ExportTarget(
            dataset_name=dataset_name,
            table_name=table_name,
            output_path=document_output_dir / filename,
            dataset=registry_meta.get(dataset_name),
        )
        exported = _export_dataset(gateway, target, spec=spec, opts=opts)
        if exported is not None:
            written.append(exported)

    manifest_path = write_dataset_manifest(
        document_output_dir,
        dataset_mapping,
        jsonl_mapping=registry.jsonl_datasets,
        parquet_mapping=registry.parquet_datasets,
        selected=list(selected.keys()),
    )
    written.append(manifest_path)

    if AUDIT_LOG_PATH is not None or AUDIT_TABLE_ENABLED:
        log.debug(
            "Export audit enabled: log_path=%s table_enabled=%s",
            AUDIT_LOG_PATH,
            AUDIT_TABLE_ENABLED,
        )

    _validate_written_exports(written, registry.by_table_key, opts)
    return written


__all__ = [
    "ExportFormat",
    "export_all_datasets",
    "export_jsonl_for_table",
    "export_parquet_for_table",
]
