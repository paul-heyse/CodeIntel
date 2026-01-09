"""Shared utilities for export operations.

This module contains common dataclasses and functions used by both
JSONL and Parquet export implementations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.errors import BuildProblemError
from codeintel.build.exports.exprs import build_export_relation_plan
from codeintel.build.schemas import iter_contracts
from codeintel.core.config.settings import ExportAuditSettings
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.datasets.manifests import dataset_manifest_path
from codeintel.core.datasets.scanning import (
    ParquetScanOptions,
    scan_parquet_dataset_with_telemetry,
)
from codeintel.core.errors.taxonomy import ErrorCode
from codeintel.core.ports.export import ExportRelation
from codeintel.core.schemas.hashing import schema_digest
from codeintel.core.validation.profiles import ValidationProfile

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pyarrow as pa

    from codeintel.build.meta.bundle import BuildMetadataBundleWriter
    from codeintel.core.gateway import BuildGateway
    from codeintel.core.schemas.contract_primitives import DatasetContract

log = logging.getLogger(__name__)

MAX_EXPORT_LIMIT = 9_223_372_036_854_775_807


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def export_problem(
    error_code: ErrorCode,
    detail: str,
    **extras: object,
) -> BuildProblemError:
    """Create an export error with structured problem details.

    Parameters
    ----------
    error_code
        Canonical error taxonomy entry.
    detail
        Detailed error message.
    **extras
        Additional context fields.

    Returns
    -------
    BuildProblemError
        BuildProblemError with structured details.
    """
    return BuildProblemError.from_error_code(error_code=error_code, detail=detail, **extras)


def log_export_error(
    error_code: ErrorCode,
    detail: str,
    **extras: object,
) -> None:
    """Log an export error with structured problem details.

    Parameters
    ----------
    error_code
        Canonical error taxonomy entry.
    detail
        Detailed error message.
    **extras
        Additional context fields.
    """
    error = BuildProblemError.from_error_code(error_code=error_code, detail=detail, **extras)
    log.error(json.dumps(error.problem_detail.to_dict()))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportAuditRecord:
    """Structured export audit record for build-first logging."""

    table_name: str
    macro: str
    rows: int
    duration_s: float
    output_path: Path


@dataclass(frozen=True)
class ExportTarget:
    """Inputs describing a dataset export request."""

    dataset_name: str
    table_name: str
    output_path: Path
    dataset: DatasetContract | None


@dataclass(frozen=True)
class ExportCallOptions:
    """Options controlling dataset selection, validation, and macro enforcement.

    Attributes
    ----------
    schemas
        Optional list of table keys to validate.
    datasets
        Optional list of dataset names to export.
    """

    validate_exports: bool = True
    schemas: list[str] | None = None
    datasets: list[str] | None = None
    validation_profile: ValidationProfile | None = None
    force_full_export: bool = False


# ---------------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------------


def validate_registry_or_raise(gateway: BuildGateway) -> None:
    """Validate dataset registry by asserting parquet manifests exist.

    Parameters
    ----------
    gateway
        BuildGateway providing access to dataset registry.

    """
    validate_dataset_manifests_or_raise(gateway)


def validate_dataset_manifests_or_raise(gateway: BuildGateway) -> None:
    """Validate that dataset manifests exist for parquet-backed datasets.

    Parameters
    ----------
    gateway
        BuildGateway providing dataset registry access.

    Raises
    ------
    ValueError
        If dataset root or snapshot metadata is missing, or manifests are absent.
    """
    dataset_root_dir = gateway.datasets.dataset_root_dir
    if dataset_root_dir is None:
        msg = "Dataset root directory is required for parquet-only exports"
        raise ValueError(msg)
    snapshot_id = getattr(gateway.config, "commit", None)
    if not isinstance(snapshot_id, str) or not snapshot_id:
        msg = "Snapshot commit is required for parquet-only exports"
        raise ValueError(msg)
    missing: list[str] = []
    for dataset in gateway.datasets.by_table_key.values():
        if dataset.is_view:
            continue
        manifest_path = dataset_manifest_path(
            dataset_root=dataset_root_dir,
            table_key=dataset.table_key,
            snapshot_id=snapshot_id,
        )
        if not manifest_path.is_file():
            missing.append(dataset.table_key)
    if missing:
        message = "Dataset manifests missing for parquet-only exports: "
        message += ", ".join(sorted(missing))
        log.warning("%s", message)


def resolve_export_snapshot(gateway: BuildGateway) -> tuple[Path, str]:
    """Resolve the dataset root + snapshot id for export reads.

    Parameters
    ----------
    gateway
        BuildGateway providing dataset registry and config access.

    Returns
    -------
    tuple[pathlib.Path, str]
        Dataset root directory and snapshot id.

    Raises
    ------
    ValueError
        If the dataset root or snapshot metadata is missing.
    """
    dataset_root_dir = gateway.datasets.dataset_root_dir
    if dataset_root_dir is None:
        msg = "Dataset root directory is required for parquet-only exports"
        raise ValueError(msg)
    snapshot_id = getattr(gateway.config, "commit", None)
    if not isinstance(snapshot_id, str) or not snapshot_id:
        msg = "Snapshot commit is required for parquet-only exports"
        raise ValueError(msg)
    return dataset_root_dir, snapshot_id


def build_export_reader(
    gateway: BuildGateway,
    table_key: str,
    *,
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE,
) -> pa.RecordBatchReader:
    """Return a RecordBatchReader for an exportable dataset snapshot.

    Parameters
    ----------
    gateway
        BuildGateway providing dataset registry access.
    table_key
        Fully qualified table key.
    batch_size
        Batch size for Arrow readers.

    Returns
    -------
    pyarrow.RecordBatchReader
        Streaming reader for the dataset snapshot.

    Raises
    ------
    FileNotFoundError
        If the dataset snapshot is missing.
    """
    dataset_root_dir, snapshot_id = resolve_export_snapshot(gateway)
    reader, telemetry = scan_parquet_dataset_with_telemetry(
        dataset_root=dataset_root_dir,
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=ParquetScanOptions(
            batch_size=batch_size,
            implicit_ordering=True,
            require_sequenced_output=True,
            metrics_enabled=True,
        ),
    )
    if telemetry is not None:
        log.debug("Export scan telemetry: %s", telemetry.to_mapping())
    if reader is None:
        msg = f"Dataset snapshot missing for {table_key}@{snapshot_id}"
        raise FileNotFoundError(msg)
    return reader


def build_export_reader_from_snapshot(
    *,
    dataset_root_dir: Path,
    snapshot_id: str,
    table_key: str,
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE,
) -> pa.RecordBatchReader:
    """Return a RecordBatchReader for an exportable dataset snapshot.

    Parameters
    ----------
    dataset_root_dir
        Root directory containing parquet dataset snapshots.
    snapshot_id
        Snapshot identifier (commit).
    table_key
        Fully qualified table key.
    batch_size
        Batch size for Arrow readers.

    Returns
    -------
    pyarrow.RecordBatchReader
        Streaming reader for the dataset snapshot.

    Raises
    ------
    FileNotFoundError
        If the dataset snapshot is missing.
    """
    reader, telemetry = scan_parquet_dataset_with_telemetry(
        dataset_root=dataset_root_dir,
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=ParquetScanOptions(
            batch_size=batch_size,
            implicit_ordering=True,
            require_sequenced_output=True,
            metrics_enabled=True,
        ),
    )
    if telemetry is not None:
        log.debug("Export scan telemetry: %s", telemetry.to_mapping())
    if reader is None:
        msg = f"Dataset snapshot missing for {table_key}@{snapshot_id}"
        raise FileNotFoundError(msg)
    return reader


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------


def resolve_dataset_table(dataset_name: str, dataset_mapping: Mapping[str, str]) -> str:
    """Resolve a dataset name to its table key.

    Parameters
    ----------
    dataset_name
        Logical dataset name.
    dataset_mapping
        Mapping of dataset name to table key.

    Returns
    -------
    str
        Fully qualified table key.

    Raises
    ------
    ValueError
        If dataset name is not in the mapping.
    """
    table = dataset_mapping.get(dataset_name)
    if table is None:
        message = f"Unknown dataset: {dataset_name}"
        raise ValueError(message)
    return table


def select_dataset_tables(
    dataset_mapping: Mapping[str, str],
    format_mapping: Mapping[str, str],
    datasets: list[str] | None,
) -> dict[str, str]:
    """Determine which dataset names and tables to export.

    Parameters
    ----------
    dataset_mapping
        Mapping of dataset name to table/view key from the gateway registry.
    format_mapping
        Mapping of table/view key to filename from the gateway registry.
    datasets
        Optional list of dataset names requested by the caller.

    Returns
    -------
    dict[str, str]
        Selected dataset name to table/view key mapping.
    """
    if datasets is None:
        return {name: table for name, table in dataset_mapping.items() if table in format_mapping}
    selected: dict[str, str] = {}
    for dataset_name in datasets:
        selected[dataset_name] = resolve_dataset_table(dataset_name, dataset_mapping)
    return selected


# ---------------------------------------------------------------------------
# Validation profile and schema digest
# ---------------------------------------------------------------------------


def resolve_validation_profile(
    options: ExportCallOptions,
    dataset: DatasetContract | None,
) -> ValidationProfile:
    """Resolve the validation profile for an export.

    Parameters
    ----------
    options
        Export options that may override the profile.
    dataset
        Dataset contract with default profile.

    Returns
    -------
    str
        Validation profile (e.g., "strict", "lenient", "schema-only").
    """
    if options.validation_profile is not None:
        return options.validation_profile
    if dataset is not None:
        return dataset.validation_profile
    return "strict"


def compute_schema_digest(dataset: DatasetContract | None) -> str | None:
    """Compute digest of the TableSchema for a dataset.

    Parameters
    ----------
    dataset
        Dataset contract to compute digest for.

    Returns
    -------
    str | None
        SHA-256 hex digest, or None if unavailable.
    """
    if dataset is None or dataset.schema is None:
        return None
    return schema_digest(dataset.schema)


# ---------------------------------------------------------------------------
def build_export_relation(
    gateway: BuildGateway,
    table_key: str,
    row_limit: int,
    row_offset: int,
) -> ExportRelation:
    """Build an export relation for a dataset table.

    Parameters
    ----------
    gateway
        BuildGateway providing connection.
    table_key
        Fully qualified table key.
    row_limit
        Maximum rows to export.
    row_offset
        Offset for pagination.

    Returns
    -------
    ExportRelation
        Export relation adapter.
    """
    relation = build_export_relation_plan(
        gateway,
        table_key,
        limit=row_limit,
        offset=row_offset,
    )
    return gateway.exports.build_export_relation(relation=relation)


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


def write_audit_entry(
    record: ExportAuditRecord,
    *,
    gateway: BuildGateway | None,
    settings: ExportAuditSettings,
    metadata_bundle: BuildMetadataBundleWriter | None = None,
) -> None:
    """Write an audit entry for an export operation.

    Parameters
    ----------
    record
        Audit record to write.
    gateway
        Storage gateway providing audit logging access.
    settings
        Export audit settings for the write.
    metadata_bundle
        Optional metadata bundle writer for build-first audit logging.
    """
    if metadata_bundle is not None:
        metadata_bundle.append_jsonl(
            "exports/export_audit.jsonl",
            {
                "dataset": record.table_name,
                "macro": record.macro,
                "rows": record.rows,
                "duration_s": record.duration_s,
                "output_path": str(record.output_path),
                "sql": None,
                "plan": None,
                "created_at": datetime.now(tz=UTC).isoformat(),
            },
            schema_version="v1",
        )
        return
    if settings.log_path is None and not settings.table_enabled:
        return
    gateway_name = type(gateway).__name__ if gateway is not None else "none"
    log.warning(
        "build.export.audit_skipped dataset=%s gateway=%s reason=missing_bundle",
        record.table_name,
        gateway_name,
    )


# ---------------------------------------------------------------------------
# Default validation schemas
# ---------------------------------------------------------------------------


def default_validation_schemas() -> list[str]:
    """Return the set of table keys that should be validated by default.

    Derived from contracts in the build.schemas contract provider.

    Returns
    -------
    list[str]
        Sorted table keys with TableSchema validation configured.
    """
    return sorted(
        contract.table_key for contract in iter_contracts() if contract.schema is not None
    )


__all__ = [
    "MAX_EXPORT_LIMIT",
    "ExportAuditRecord",
    "ExportCallOptions",
    "ExportTarget",
    "build_export_reader",
    "build_export_reader_from_snapshot",
    "build_export_relation",
    "compute_schema_digest",
    "default_validation_schemas",
    "export_problem",
    "log_export_error",
    "resolve_dataset_table",
    "resolve_validation_profile",
    "select_dataset_tables",
    "validate_dataset_manifests_or_raise",
    "validate_registry_or_raise",
    "write_audit_entry",
]
