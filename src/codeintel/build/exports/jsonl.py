"""JSON/JSONL exporters for the CodeIntel metadata warehouse."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from codeintel.build.exports.common import (
    MAX_EXPORT_LIMIT,
    AuditRecord,
    ExportCallOptions,
    ExportError,
    ExportTarget,
    build_export_relation,
    compute_schema_digest,
    default_validation_schemas,
    get_row_count,
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
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@runtime_checkable
class _SupportsIsoformat(Protocol):
    def isoformat(self) -> str: ...


def _default_serializer(obj: object) -> object:
    """Serialize objects for JSON output.

    Parameters
    ----------
    obj
        Object to serialize.

    Returns
    -------
    object
        JSON-serializable representation.

    Raises
    ------
    TypeError
        If object is not serializable.
    """
    if isinstance(obj, _SupportsIsoformat):
        return obj.isoformat()
    message = f"Type {type(obj)} is not JSON serializable"
    raise TypeError(message)


def export_jsonl_for_table(
    gateway: StorageGateway,
    table_name: str,
    output_path: Path,
    *,
    serializer: Callable[[object], object] | None = None,
) -> None:
    """Export a single DuckDB table to JSONL.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection.
    table_name
        Fully qualified table name (schema.table) to export.
    output_path
        Destination path for the JSONL file.
    serializer
        Custom JSON serializer for complex types.

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
    default = serializer or _default_serializer
    start = perf_counter()
    rel = build_export_relation(gateway, table_name, MAX_EXPORT_LIMIT, 0)
    macro_name = "ibis_export"
    df = rel.df()
    records = df.to_dict(orient="records")
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=default))
            f.write("\n")
    duration = perf_counter() - start
    rows = len(records)
    write_audit_entry(
        AuditRecord(
            table_name=table_name,
            macro=macro_name,
            rows=rows,
            duration_s=duration,
            output_path=output_path,
        ),
        con=gateway.con,
    )
    log.debug(
        "Exported %s rows for %s via Ibis export in %.3fs",
        rows,
        table_name,
        duration,
    )


def export_dataset_to_jsonl(
    gateway: StorageGateway,
    dataset_name: str,
    output_dir: Path,
) -> Path:
    """Export a dataset resolved through the dataset registry to JSONL.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection.
    dataset_name
        Logical dataset name to export (e.g., ``function_profile``).
    output_dir
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
    export_jsonl_for_table(gateway, table_name, output_path)
    return output_path


def _export_dataset_jsonl(
    gateway: StorageGateway,
    target: ExportTarget,
    *,
    opts: ExportCallOptions,
) -> Path | None:
    """Export a single dataset to JSONL with incremental support.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection.
    target
        Export target specification.
    opts
        Export call options.

    Returns
    -------
    Path | None
        Path to exported file, or None if skipped.
    """
    if target.dataset is not None and not target.dataset.capabilities()["can_export_jsonl"]:
        log.warning("Skipping dataset %s; JSONL export not supported", target.dataset_name)
        return None
    validation_profile = resolve_validation_profile(opts, target.dataset)
    schema_digest = compute_schema_digest(target.dataset)
    marker = read_incremental_marker(target.output_path)
    current_row_count: int | None = None
    if target.dataset is None or not target.dataset.is_view:
        current_row_count = get_row_count(gateway, target.table_name)
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
        export_jsonl_for_table(gateway, target.table_name, target.output_path)
        data_hash = compute_file_hash(target.output_path)
        completed_at = datetime.now(UTC)
        final_row_count = (
            current_row_count
            if current_row_count is not None
            else get_row_count(gateway, target.table_name)
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
    """Validate written export files against schemas.

    Parameters
    ----------
    written
        List of written file paths.
    registry_meta
        Dataset contract metadata by name.
    opts
        Export call options.

    Raises
    ------
    ExportError
        If validation fails for a schema in strict mode.
    """
    if not opts.validate_exports:
        return
    schema_list = opts.schemas or default_validation_schemas()
    for schema_name in schema_list:
        matching = [p for p in written if p.name.startswith(schema_name)]
        if not matching:
            continue
        ds = registry_meta.get(schema_name)
        if ds is None or ds.json_schema_id is None:
            log.info("Skipping validation for %s; no JSON Schema configured", schema_name)
            continue
        profile = resolve_validation_profile(opts, ds)
        exit_code = validate_export_files(schema_name, matching)
        if exit_code != 0 and profile == "lenient":
            log_export_error(
                code="export.validation_failed",
                title="Export validation failed",
                detail=f"Validation failed for schema {schema_name}",
                schema=schema_name,
                files=[str(p) for p in matching],
            )
            continue
        if exit_code != 0:
            msg = f"Validation failed for schema {schema_name}"
            log_export_error(
                code="export.validation_failed",
                title="Export validation failed",
                detail=msg,
                schema=schema_name,
                files=[str(p) for p in matching],
            )
            raise ExportError(msg)


def export_all_jsonl(
    gateway: StorageGateway,
    document_output_dir: Path,
    *,
    options: ExportCallOptions | None = None,
) -> list[Path]:
    """Export configured datasets to JSONL files under `Document Output/`.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection.
    document_output_dir
        Target directory where JSONL artifacts are written.
    options
        Export options controlling dataset selection and validation.

    Returns
    -------
    list[Path]
        List of written file paths.
    """
    opts = options or ExportCallOptions()
    document_output_dir = document_output_dir.resolve()
    document_output_dir.mkdir(parents=True, exist_ok=True)

    validate_registry_or_raise(gateway)
    dataset_mapping = gateway.datasets.mapping
    jsonl_mapping = gateway.datasets.jsonl_mapping or {}
    parquet_mapping = gateway.datasets.parquet_mapping or {}
    registry_meta = gateway.datasets.meta or {}
    selected = select_dataset_tables(dataset_mapping, jsonl_mapping, opts.datasets)
    missing_tables = set(jsonl_mapping) - set(dataset_mapping.values())
    for table_name in sorted(missing_tables):
        log.warning("Skipping %s; table not present in dataset registry", table_name)

    written: list[Path] = []

    for dataset_name, table_name in sorted(selected.items()):
        filename = jsonl_mapping.get(table_name, f"{dataset_name}.jsonl")
        target = ExportTarget(
            dataset_name=dataset_name,
            table_name=table_name,
            output_path=document_output_dir / filename,
            dataset=registry_meta.get(dataset_name),
        )
        exported = _export_dataset_jsonl(gateway, target, opts=opts)
        if exported is not None:
            written.append(exported)

    manifest_path = write_dataset_manifest(
        document_output_dir,
        dataset_mapping,
        jsonl_mapping=jsonl_mapping,
        parquet_mapping=parquet_mapping,
        selected=list(selected.keys()),
    )
    written.append(manifest_path)

    _validate_written_exports(written, registry_meta, opts)
    return written


def export_repo_map_json(
    gateway: StorageGateway,
    document_output_dir: Path,
    *,
    format_output: Literal["json", "jsonl"] = "json",
) -> Path:
    """Export the repo_map table as JSON or JSONL.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection.
    document_output_dir
        Target directory where the export artifact is written.
    format_output
        Output format; "json" produces a single JSON array, "jsonl"
        produces newline-delimited JSON records.

    Returns
    -------
    Path
        Path to the written file.
    """
    document_output_dir = document_output_dir.resolve()
    document_output_dir.mkdir(parents=True, exist_ok=True)
    table_name = "core.repo_map"
    rel = build_export_relation(gateway, table_name, MAX_EXPORT_LIMIT, 0)
    df = rel.df()
    records = df.to_dict(orient="records")
    if format_output == "json":
        output_path = document_output_dir / "repo_map.json"
        output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    else:
        output_path = document_output_dir / "repo_map.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record))
                f.write("\n")
    return output_path


__all__ = [
    "export_all_jsonl",
    "export_dataset_to_jsonl",
    "export_jsonl_for_table",
    "export_repo_map_json",
]
