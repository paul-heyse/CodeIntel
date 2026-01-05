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
from typing import TYPE_CHECKING, Literal, cast

from codeintel.build.errors import BuildProblemError
from codeintel.build.exports.common import (
    ExportAuditRecord,
    ExportCallOptions,
    ExportTarget,
    build_export_reader,
    build_export_reader_from_snapshot,
    compute_schema_digest,
    default_validation_schemas,
    log_export_error,
    resolve_export_snapshot,
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
    write_jsonl_reader,
    write_parquet_reader,
)
from codeintel.build.schemas import iter_contracts
from codeintel.core.config.settings import ExportAuditSettings
from codeintel.core.datasets.manifests import dataset_manifest_path, load_dataset_manifest
from codeintel.core.errors.schema import SCHEMA_VALIDATION_FAILED
from codeintel.core.exports.formats import normalize_export_format, suffix_for_export_format

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from codeintel.build.meta.bundle import BuildMetadataBundleWriter
    from codeintel.core.gateway import BuildGateway, DatasetRegistryProtocol
    from codeintel.core.schemas.contract_primitives import DatasetContract

log = logging.getLogger(__name__)

ExportFormat = Literal["jsonl", "parquet"]
_BUILD_EXPORT_FORMATS = {"jsonl", "parquet"}

_EXPORT_RECORD_BATCH_SIZE = 10_000


@dataclass(frozen=True, slots=True)
class _ExportFormatSpec:
    format: ExportFormat
    mapping: Mapping[str, str]
    can_export_capability_key: str
    extension: str
    write_table: Callable[
        [BuildGateway, str, Path, ExportAuditSettings, BuildMetadataBundleWriter | None], int
    ]


@dataclass(frozen=True, slots=True)
class ExportRunConfig:
    """Configuration inputs for export runs."""

    settings: ExportAuditSettings
    options: ExportCallOptions | None = None
    metadata_bundle: BuildMetadataBundleWriter | None = None


@dataclass(frozen=True, slots=True)
class _ExportRunContext:
    spec: _ExportFormatSpec
    opts: ExportCallOptions
    settings: ExportAuditSettings
    metadata_bundle: BuildMetadataBundleWriter | None


@dataclass(frozen=True, slots=True)
class _SnapshotRunContext:
    opts: ExportCallOptions
    settings: ExportAuditSettings
    metadata_bundle: BuildMetadataBundleWriter | None


@dataclass(frozen=True, slots=True)
class _ExportPlan:
    output_dir: Path
    registry: DatasetRegistryProtocol
    dataset_mapping: dict[str, str]
    selected: dict[str, str]
    spec: _ExportFormatSpec
    context: _ExportRunContext


@dataclass(frozen=True, slots=True)
class SnapshotExportSource:
    """Snapshot pointer for Arrow-backed export reads."""

    dataset_root_dir: Path
    snapshot_id: str


@dataclass(frozen=True, slots=True)
class _SnapshotExportPlan:
    source: SnapshotExportSource
    output_dir: Path
    dataset_mapping: dict[str, str]
    by_name: dict[str, DatasetContract]
    by_table_key: dict[str, DatasetContract]
    selected: dict[str, str]
    format_mapping: dict[str, str]
    extension: str
    capability_key: str
    normalized_format: ExportFormat
    contracts: tuple[DatasetContract, ...]
    context: _SnapshotRunContext


@dataclass(frozen=True, slots=True)
class _ExportDecision:
    validation_profile: str
    schema_digest: str | None
    current_row_count: int | None
    should_skip: bool
    skip_path: Path | None


@dataclass(frozen=True, slots=True)
class _ExportWriteResult:
    rows_written: int | None
    data_hash: str
    started_at: datetime
    completed_at: datetime
    final_row_count: int | None


def export_jsonl_for_table(
    gateway: BuildGateway,
    table_key: str,
    output_path: Path,
    settings: ExportAuditSettings,
    metadata_bundle: BuildMetadataBundleWriter | None = None,
) -> int:
    """Export a dataset snapshot to JSONL.

    Parameters
    ----------
    gateway
        Storage gateway providing dataset registry access.
    table_key
        Fully qualified table key to export (schema.table).
    output_path
        Output JSONL path.
    settings
        Export audit settings.
    metadata_bundle
        Optional metadata bundle writer for build-first audit logging.

    Returns
    -------
    int
        Number of rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    rows_written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        reader = build_export_reader(
            gateway,
            table_key,
            batch_size=_EXPORT_RECORD_BATCH_SIZE,
        )
        rows_written = write_jsonl_reader(handle, reader=reader)
    duration = perf_counter() - start
    write_audit_entry(
        ExportAuditRecord(
            table_name=table_key,
            macro="duckdb_relation",
            rows=rows_written,
            duration_s=duration,
            output_path=output_path,
        ),
        gateway=gateway,
        settings=settings,
        metadata_bundle=metadata_bundle,
    )
    return rows_written


def export_parquet_for_table(
    gateway: BuildGateway,
    table_key: str,
    output_path: Path,
    settings: ExportAuditSettings,
    metadata_bundle: BuildMetadataBundleWriter | None = None,
) -> int:
    """Export a dataset snapshot to Parquet.

    Parameters
    ----------
    gateway
        Storage gateway providing dataset registry access.
    table_key
        Fully qualified table key to export (schema.table).
    output_path
        Output Parquet path.
    settings
        Export audit settings.
    metadata_bundle
        Optional metadata bundle writer for build-first audit logging.

    Returns
    -------
    int
        Number of rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    reader = build_export_reader(
        gateway,
        table_key,
        batch_size=_EXPORT_RECORD_BATCH_SIZE,
    )
    dictionary_encode, dictionary_columns = _dictionary_options_for_export(gateway, table_key)
    rows_written = write_parquet_reader(
        reader=reader,
        output_path=output_path,
        dictionary_encode=dictionary_encode,
        dictionary_columns=dictionary_columns,
    )
    duration = perf_counter() - start
    write_audit_entry(
        ExportAuditRecord(
            table_name=table_key,
            macro="duckdb_relation",
            rows=rows_written,
            duration_s=duration,
            output_path=output_path,
        ),
        gateway=gateway,
        settings=settings,
        metadata_bundle=metadata_bundle,
    )
    return rows_written


def export_jsonl_for_table_from_snapshot(
    *,
    source: SnapshotExportSource,
    target: ExportTarget,
    context: _SnapshotRunContext,
) -> int:
    """Export a dataset snapshot to JSONL without a storage gateway.

    Parameters
    ----------
    source
        Dataset snapshot location.
    target
        Export target describing the dataset and output path.
    context
        Snapshot export context including audit settings.

    Returns
    -------
    int
        Number of rows written.
    """
    target.output_path.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    reader = build_export_reader_from_snapshot(
        dataset_root_dir=source.dataset_root_dir,
        snapshot_id=source.snapshot_id,
        table_key=target.table_name,
        batch_size=_EXPORT_RECORD_BATCH_SIZE,
    )
    with target.output_path.open("w", encoding="utf-8") as handle:
        rows_written = write_jsonl_reader(handle=handle, reader=reader)
    duration = perf_counter() - start
    write_audit_entry(
        ExportAuditRecord(
            table_name=target.table_name,
            macro="parquet_snapshot",
            rows=rows_written,
            duration_s=duration,
            output_path=target.output_path,
        ),
        gateway=None,
        settings=context.settings,
        metadata_bundle=context.metadata_bundle,
    )
    return rows_written


def export_parquet_for_table_from_snapshot(
    *,
    source: SnapshotExportSource,
    target: ExportTarget,
    context: _SnapshotRunContext,
    manifest: object | None = None,
) -> int:
    """Export a dataset snapshot to Parquet without a storage gateway.

    Parameters
    ----------
    source
        Dataset snapshot location.
    target
        Export target describing the dataset and output path.
    context
        Snapshot export context including audit settings.
    manifest
        Optional dataset manifest used for dictionary encoding hints.

    Returns
    -------
    int
        Number of rows written.
    """
    target.output_path.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    reader = build_export_reader_from_snapshot(
        dataset_root_dir=source.dataset_root_dir,
        snapshot_id=source.snapshot_id,
        table_key=target.table_name,
        batch_size=_EXPORT_RECORD_BATCH_SIZE,
    )
    dictionary_encode, dictionary_columns = _dictionary_options_from_manifest(manifest)
    rows_written = write_parquet_reader(
        reader=reader,
        output_path=target.output_path,
        dictionary_encode=dictionary_encode,
        dictionary_columns=dictionary_columns,
    )
    duration = perf_counter() - start
    write_audit_entry(
        ExportAuditRecord(
            table_name=target.table_name,
            macro="parquet_snapshot",
            rows=rows_written,
            duration_s=duration,
            output_path=target.output_path,
        ),
        gateway=None,
        settings=context.settings,
        metadata_bundle=context.metadata_bundle,
    )
    return rows_written


def _dictionary_options_for_export(
    gateway: BuildGateway,
    table_key: str,
) -> tuple[bool, tuple[str, ...] | None]:
    dataset_root_dir, snapshot_id = resolve_export_snapshot(gateway)
    manifest = load_dataset_manifest(
        dataset_root=dataset_root_dir,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    return _dictionary_options_from_manifest(manifest)


def _dictionary_options_from_manifest(
    manifest: object | None,
) -> tuple[bool, tuple[str, ...] | None]:
    if manifest is None:
        return False, None
    extras = getattr(manifest, "extras", None)
    if not isinstance(extras, dict):
        return False, None
    write_settings = _coerce_mapping(extras.get("write_settings"))
    inferred_settings = _coerce_mapping(extras.get("inferred_settings"))
    columns = _read_str_list(write_settings.get("dictionary_encode_columns"))
    if not columns:
        columns = _read_str_list(inferred_settings.get("dictionary_encode_columns"))
    if columns:
        return True, tuple(columns)
    dictionary_encode = _read_bool(write_settings.get("dictionary_encode"))
    if dictionary_encode:
        return True, None
    return False, None


def _read_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _read_str_list(value: object) -> list[str] | None:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return None


def _coerce_mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): val for key, val in value.items()}
    return {}


def _format_spec(gateway: BuildGateway, fmt: ExportFormat) -> _ExportFormatSpec:
    if fmt == "jsonl":
        return _ExportFormatSpec(
            format="jsonl",
            mapping=gateway.datasets.jsonl_datasets,
            can_export_capability_key="can_export_jsonl",
            extension=suffix_for_export_format(fmt),
            write_table=export_jsonl_for_table,
        )
    return _ExportFormatSpec(
        format="parquet",
        mapping=gateway.datasets.parquet_datasets,
        can_export_capability_key="can_export_parquet",
        extension=suffix_for_export_format(fmt),
        write_table=export_parquet_for_table,
    )


def _normalize_build_format(fmt: str) -> ExportFormat:
    normalized = normalize_export_format(fmt)
    if normalized not in _BUILD_EXPORT_FORMATS:
        msg = f"Unsupported export format for build: {fmt}"
        raise ValueError(msg)
    return cast("ExportFormat", normalized)


def _build_export_plan(
    *,
    gateway: BuildGateway,
    document_output_dir: Path,
    fmt: ExportFormat,
    run_config: ExportRunConfig,
) -> _ExportPlan:
    opts = run_config.options or ExportCallOptions()
    output_dir = document_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_format = _normalize_build_format(fmt)
    validate_registry_or_raise(gateway)
    registry = gateway.datasets
    dataset_mapping = {name: contract.table_key for name, contract in registry.by_name.items()}
    spec = _format_spec(gateway, normalized_format)
    context = _ExportRunContext(
        spec=spec,
        opts=opts,
        settings=run_config.settings,
        metadata_bundle=run_config.metadata_bundle,
    )
    selected = select_dataset_tables(dataset_mapping, spec.mapping, opts.datasets)
    _log_missing_tables(spec.mapping, dataset_mapping)
    return _ExportPlan(
        output_dir=output_dir,
        registry=registry,
        dataset_mapping=dataset_mapping,
        selected=selected,
        spec=spec,
        context=context,
    )


def _build_snapshot_export_plan(
    *,
    source: SnapshotExportSource,
    document_output_dir: Path,
    fmt: ExportFormat,
    run_config: ExportRunConfig,
    contracts: Sequence[DatasetContract] | None,
) -> _SnapshotExportPlan:
    opts = run_config.options or ExportCallOptions()
    output_dir = document_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_format = _normalize_build_format(fmt)
    contract_list = tuple(contracts) if contracts is not None else tuple(iter_contracts())
    _validate_contract_manifests(
        dataset_root_dir=source.dataset_root_dir,
        snapshot_id=source.snapshot_id,
        contracts=contract_list,
    )
    dataset_mapping, by_name, by_table_key = _contract_mappings(contract_list)
    format_mapping, extension, capability_key = _format_mapping_for_contracts(
        contract_list,
        normalized_format,
    )
    selected = select_dataset_tables(dataset_mapping, format_mapping, opts.datasets)
    _log_missing_tables(format_mapping, dataset_mapping)
    context = _SnapshotRunContext(
        settings=run_config.settings,
        opts=opts,
        metadata_bundle=run_config.metadata_bundle,
    )
    return _SnapshotExportPlan(
        source=source,
        output_dir=output_dir,
        dataset_mapping=dataset_mapping,
        by_name=by_name,
        by_table_key=by_table_key,
        selected=selected,
        format_mapping=format_mapping,
        extension=extension,
        capability_key=capability_key,
        normalized_format=normalized_format,
        contracts=contract_list,
        context=context,
    )


def _log_missing_tables(mapping: Mapping[str, str], dataset_mapping: dict[str, str]) -> None:
    missing_tables = set(mapping) - set(dataset_mapping.values())
    for table_name in sorted(missing_tables):
        log.warning("Skipping %s; table not present in dataset registry", table_name)


def _contract_mappings(
    contracts: Sequence[DatasetContract],
) -> tuple[dict[str, str], dict[str, DatasetContract], dict[str, DatasetContract]]:
    dataset_mapping: dict[str, str] = {}
    by_name: dict[str, DatasetContract] = {}
    by_table_key: dict[str, DatasetContract] = {}
    for contract in contracts:
        dataset_mapping[contract.name] = contract.table_key
        by_name[contract.name] = contract
        by_table_key[contract.table_key] = contract
    return dataset_mapping, by_name, by_table_key


def _format_mapping_for_contracts(
    contracts: Sequence[DatasetContract],
    fmt: ExportFormat,
) -> tuple[dict[str, str], str, str]:
    mapping: dict[str, str] = {}
    extension = suffix_for_export_format(fmt)
    capability_key = "can_export_jsonl" if fmt == "jsonl" else "can_export_parquet"
    for contract in contracts:
        filename = contract.jsonl_filename if fmt == "jsonl" else contract.parquet_filename
        if isinstance(filename, str) and filename:
            mapping[contract.table_key] = filename
    return mapping, extension, capability_key


def _validate_contract_manifests(
    *,
    dataset_root_dir: Path,
    snapshot_id: str,
    contracts: Sequence[DatasetContract],
) -> None:
    missing: list[str] = []
    for contract in contracts:
        if contract.is_view:
            continue
        manifest_path = dataset_manifest_path(
            dataset_root=dataset_root_dir,
            table_key=contract.table_key,
            snapshot_id=snapshot_id,
        )
        if not manifest_path.is_file():
            missing.append(contract.table_key)
    if missing:
        message = "Dataset manifests missing for parquet-only exports: "
        message += ", ".join(sorted(missing))
        raise ValueError(message)


def _current_row_count(gateway: BuildGateway, table_key: str) -> int | None:
    dataset_root_dir, snapshot_id = resolve_export_snapshot(gateway)
    manifest = load_dataset_manifest(
        dataset_root=dataset_root_dir,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    return manifest.row_count if manifest is not None else None


def _build_export_decision(
    *,
    gateway: BuildGateway,
    target: ExportTarget,
    opts: ExportCallOptions,
) -> _ExportDecision:
    validation_profile = resolve_validation_profile(opts, target.dataset)
    schema_digest = compute_schema_digest(target.dataset)
    marker = read_incremental_marker(target.output_path)
    current_row_count = _current_row_count(gateway, target.table_name)
    criteria = SkipCriteria(
        row_count=current_row_count,
        schema_version=target.dataset.schema_version if target.dataset else None,
        validation_profile=validation_profile,
        schema_digest=schema_digest,
        force_full_export=opts.force_full_export,
    )
    should_skip = should_skip_export(marker, criteria)
    skip_path = target.output_path if should_skip and target.output_path.exists() else None
    return _ExportDecision(
        validation_profile=validation_profile,
        schema_digest=schema_digest,
        current_row_count=current_row_count,
        should_skip=should_skip,
        skip_path=skip_path,
    )


def _perform_export_write(
    *,
    gateway: BuildGateway,
    target: ExportTarget,
    context: _ExportRunContext,
    decision: _ExportDecision,
) -> _ExportWriteResult:
    started_at = datetime.now(UTC)
    rows_written = context.spec.write_table(
        gateway,
        target.table_name,
        target.output_path,
        context.settings,
        context.metadata_bundle,
    )
    data_hash = compute_file_hash(target.output_path)
    completed_at = datetime.now(UTC)
    final_row_count = rows_written if rows_written is not None else decision.current_row_count
    return _ExportWriteResult(
        rows_written=rows_written,
        data_hash=data_hash,
        started_at=started_at,
        completed_at=completed_at,
        final_row_count=final_row_count,
    )


def _validate_written_exports(
    written: list[Path],
    registry_by_table_key: Mapping[str, DatasetContract],
    opts: ExportCallOptions,
    gateway: BuildGateway | None,
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
        if dataset.schema is None:
            log.info("Skipping validation for %s; no TableSchema configured", table_key)
            continue
        profile = resolve_validation_profile(opts, dataset)
        exit_code = validate_export_files(
            table_key,
            matching,
            dataset_name=dataset_name,
            gateway=gateway,
            validation_profile=profile,
        )
        if exit_code != 0 and profile == "lenient":
            log_export_error(
                SCHEMA_VALIDATION_FAILED,
                f"Validation failed for schema {table_key}",
                table_key=table_key,
                files=[str(p) for p in matching],
            )
            continue
        if exit_code != 0:
            msg = f"Validation failed for schema {table_key}"
            log_export_error(
                SCHEMA_VALIDATION_FAILED,
                msg,
                table_key=table_key,
                files=[str(p) for p in matching],
            )
            error = BuildProblemError.from_error_code(
                error_code=SCHEMA_VALIDATION_FAILED,
                detail=msg,
                table_key=table_key,
                files=[str(p) for p in matching],
            )
            raise error


def _export_dataset(
    gateway: BuildGateway,
    target: ExportTarget,
    *,
    context: _ExportRunContext,
) -> Path | None:
    spec = context.spec
    if target.dataset is not None:
        caps = target.dataset.capabilities()
        if not caps.get(spec.can_export_capability_key, False):
            log.warning(
                "Skipping dataset %s; %s export not supported",
                target.dataset_name,
                spec.format,
            )
            return None

    decision = _build_export_decision(
        gateway=gateway,
        target=target,
        opts=context.opts,
    )
    if decision.should_skip:
        return decision.skip_path

    try:
        result = _perform_export_write(
            gateway=gateway,
            target=target,
            context=context,
            decision=decision,
        )
    except (OSError, ValueError, TypeError) as exc:
        log.warning(
            "Failed to export dataset %s (%s) to %s: %s",
            target.dataset_name,
            target.table_name,
            target.output_path,
            exc,
        )
        return None

    row_count = (
        result.final_row_count
        if result.final_row_count is not None
        else (result.rows_written if result.rows_written is not None else 0)
    )
    manifest_payload = ExportManifestData(
        dataset=target.dataset_name,
        artifact=target.output_path.name,
        schema_id=target.dataset.json_schema_id if target.dataset else None,
        schema_version=target.dataset.schema_version if target.dataset else None,
        schema_digest=decision.schema_digest,
        validation_profile=decision.validation_profile,
        row_count=row_count,
        data_hash=result.data_hash,
        started_at=result.started_at.isoformat(),
        completed_at=result.completed_at.isoformat(),
    )
    write_per_dataset_manifest(target.output_path, manifest_payload)
    write_incremental_marker(
        target.output_path,
        IncrementalMarker(
            dataset=target.dataset_name,
            row_count=row_count,
            schema_version=target.dataset.schema_version if target.dataset else None,
            validation_profile=decision.validation_profile,
            schema_digest=decision.schema_digest,
        ),
    )
    return target.output_path


def _export_dataset_from_snapshot(
    *,
    source: SnapshotExportSource,
    target: ExportTarget,
    fmt: ExportFormat,
    context: _SnapshotRunContext,
) -> Path | None:
    caps = target.dataset.capabilities() if target.dataset is not None else {}
    capability_key = "can_export_jsonl" if fmt == "jsonl" else "can_export_parquet"
    if caps and not caps.get(capability_key, False):
        log.warning(
            "Skipping dataset %s; %s export not supported",
            target.dataset_name,
            fmt,
        )
        return None

    manifest = load_dataset_manifest(
        dataset_root=source.dataset_root_dir,
        table_key=target.table_name,
        snapshot_id=source.snapshot_id,
    )
    decision = _build_export_decision_from_snapshot(
        target=target,
        opts=context.opts,
        manifest=manifest,
    )
    if decision.should_skip:
        return decision.skip_path

    try:
        result = _perform_export_write_from_snapshot(
            source=source,
            target=target,
            fmt=fmt,
            context=context,
            manifest=manifest,
        )
    except (OSError, ValueError, TypeError) as exc:
        log.warning(
            "Failed to export dataset %s (%s) to %s: %s",
            target.dataset_name,
            target.table_name,
            target.output_path,
            exc,
        )
        return None

    row_count = (
        result.final_row_count
        if result.final_row_count is not None
        else (result.rows_written if result.rows_written is not None else 0)
    )
    manifest_payload = ExportManifestData(
        dataset=target.dataset_name,
        artifact=target.output_path.name,
        schema_id=target.dataset.json_schema_id if target.dataset else None,
        schema_version=target.dataset.schema_version if target.dataset else None,
        schema_digest=decision.schema_digest,
        validation_profile=decision.validation_profile,
        row_count=row_count,
        data_hash=result.data_hash,
        started_at=result.started_at.isoformat(),
        completed_at=result.completed_at.isoformat(),
    )
    write_per_dataset_manifest(target.output_path, manifest_payload)
    write_incremental_marker(
        target.output_path,
        IncrementalMarker(
            dataset=target.dataset_name,
            row_count=row_count,
            schema_version=target.dataset.schema_version if target.dataset else None,
            validation_profile=decision.validation_profile,
            schema_digest=decision.schema_digest,
        ),
    )
    return target.output_path


def _build_export_decision_from_snapshot(
    *,
    target: ExportTarget,
    opts: ExportCallOptions,
    manifest: object | None,
) -> _ExportDecision:
    validation_profile = resolve_validation_profile(opts, target.dataset)
    schema_digest = compute_schema_digest(target.dataset)
    marker = read_incremental_marker(target.output_path)
    current_row_count = getattr(manifest, "row_count", None)
    criteria = SkipCriteria(
        row_count=current_row_count if isinstance(current_row_count, int) else None,
        schema_version=target.dataset.schema_version if target.dataset else None,
        validation_profile=validation_profile,
        schema_digest=schema_digest,
        force_full_export=opts.force_full_export,
    )
    should_skip = should_skip_export(marker, criteria)
    skip_path = target.output_path if should_skip and target.output_path.exists() else None
    return _ExportDecision(
        validation_profile=validation_profile,
        schema_digest=schema_digest,
        current_row_count=current_row_count if isinstance(current_row_count, int) else None,
        should_skip=should_skip,
        skip_path=skip_path,
    )


def _perform_export_write_from_snapshot(
    *,
    source: SnapshotExportSource,
    target: ExportTarget,
    fmt: ExportFormat,
    context: _SnapshotRunContext,
    manifest: object | None,
) -> _ExportWriteResult:
    started_at = datetime.now(UTC)
    if fmt == "jsonl":
        rows_written = export_jsonl_for_table_from_snapshot(
            source=source,
            target=target,
            context=context,
        )
    else:
        rows_written = export_parquet_for_table_from_snapshot(
            source=source,
            target=target,
            context=context,
            manifest=manifest,
        )
    data_hash = compute_file_hash(target.output_path)
    completed_at = datetime.now(UTC)
    final_row_count = (
        rows_written if rows_written is not None else getattr(manifest, "row_count", None)
    )
    return _ExportWriteResult(
        rows_written=rows_written,
        data_hash=data_hash,
        started_at=started_at,
        completed_at=completed_at,
        final_row_count=final_row_count if isinstance(final_row_count, int) else None,
    )


def export_all_datasets(
    gateway: BuildGateway,
    document_output_dir: Path,
    *,
    fmt: ExportFormat,
    run_config: ExportRunConfig,
) -> list[Path]:
    """Export configured datasets to a given format under the document output directory.

    Parameters
    ----------
    gateway
        Storage gateway providing dataset registry access.
    document_output_dir
        Root directory under which dataset artifacts are written.
    fmt
        Export format ("jsonl" or "parquet").
    run_config
        Export settings, selection options, and optional metadata bundle.

    Returns
    -------
    list[Path]
        Paths to written dataset artifacts and the top-level manifest.
    """
    plan = _build_export_plan(
        gateway=gateway,
        document_output_dir=document_output_dir,
        fmt=fmt,
        run_config=run_config,
    )
    written: list[Path] = []
    for dataset_name, table_name in sorted(plan.selected.items()):
        filename = plan.spec.mapping.get(table_name, f"{dataset_name}{plan.spec.extension}")
        target = ExportTarget(
            dataset_name=dataset_name,
            table_name=table_name,
            output_path=plan.output_dir / filename,
            dataset=plan.registry.by_name.get(dataset_name),
        )
        exported = _export_dataset(
            gateway,
            target,
            context=plan.context,
        )
        if exported is not None:
            written.append(exported)

    manifest_path = write_dataset_manifest(
        plan.output_dir,
        plan.dataset_mapping,
        jsonl_mapping=plan.registry.jsonl_datasets,
        parquet_mapping=plan.registry.parquet_datasets,
        selected=list(plan.selected.keys()),
    )
    written.append(manifest_path)

    if gateway.exports.audit_enabled(plan.context.settings):
        log.debug(
            "Export audit enabled: log_path=%s table_enabled=%s",
            plan.context.settings.log_path,
            plan.context.settings.table_enabled,
        )

    _validate_written_exports(written, plan.registry.by_table_key, plan.context.opts, gateway)
    return written


def export_all_datasets_from_snapshot(
    *,
    source: SnapshotExportSource,
    document_output_dir: Path,
    fmt: ExportFormat,
    run_config: ExportRunConfig,
    contracts: Sequence[DatasetContract] | None = None,
) -> list[Path]:
    """Export datasets from Arrow snapshots without a storage gateway.

    Parameters
    ----------
    source
        Dataset snapshot location.
    document_output_dir
        Output directory for export artifacts.
    fmt
        Export format ("jsonl" or "parquet").
    run_config
        Export settings, selection options, and optional metadata bundle.
    contracts
        Optional override list of dataset contracts.

    Returns
    -------
    list[pathlib.Path]
        Paths to written dataset artifacts and the top-level manifest.
    """
    plan = _build_snapshot_export_plan(
        source=source,
        document_output_dir=document_output_dir,
        fmt=fmt,
        run_config=run_config,
        contracts=contracts,
    )
    written: list[Path] = []
    for dataset_name, table_name in sorted(plan.selected.items()):
        dataset = plan.by_name.get(dataset_name)
        if dataset is None:
            continue
        caps = dataset.capabilities()
        if not caps.get(plan.capability_key, False):
            log.warning(
                "Skipping dataset %s; %s export not supported",
                dataset_name,
                plan.normalized_format,
            )
            continue
        filename = plan.format_mapping.get(table_name, f"{dataset_name}{plan.extension}")
        target = ExportTarget(
            dataset_name=dataset_name,
            table_name=table_name,
            output_path=plan.output_dir / filename,
            dataset=dataset,
        )
        exported = _export_dataset_from_snapshot(
            source=plan.source,
            target=target,
            fmt=plan.normalized_format,
            context=plan.context,
        )
        if exported is not None:
            written.append(exported)

    manifest_path = write_dataset_manifest(
        plan.output_dir,
        plan.dataset_mapping,
        jsonl_mapping=_format_mapping_for_contracts(plan.contracts, "jsonl")[0],
        parquet_mapping=_format_mapping_for_contracts(plan.contracts, "parquet")[0],
        selected=list(plan.selected.keys()),
    )
    written.append(manifest_path)
    _validate_written_exports(written, plan.by_table_key, plan.context.opts, gateway=None)
    return written


__all__ = [
    "ExportFormat",
    "ExportRunConfig",
    "SnapshotExportSource",
    "export_all_datasets",
    "export_all_datasets_from_snapshot",
    "export_jsonl_for_table",
    "export_jsonl_for_table_from_snapshot",
    "export_parquet_for_table",
    "export_parquet_for_table_from_snapshot",
]
