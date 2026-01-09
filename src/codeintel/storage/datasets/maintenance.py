"""Arrow dataset maintenance utilities (rewrite, compact, vacuum, verify)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.finalize_ops import (
    FinalizeDedupe,
    FinalizeResult,
    FinalizeSpec,
    finalize_table,
)
from codeintel.core.columnar.normalization import normalize_table_for_compute
from codeintel.core.constants import DEFAULT_ARROW_PROVENANCE_COLUMNS
from codeintel.core.datasets.arrow_store import (
    ArrowDatasetWriteOptions,
    ExistingDataBehavior,
    write_dataset,
)
from codeintel.core.datasets.manifests import (
    dataset_manifest_path,
    read_dataset_manifest,
)
from codeintel.core.datasets.scanning import (
    ParquetScanOptions,
    scan_parquet_dataset_with_telemetry,
)
from codeintel.core.manifests import ArrowDatasetManifest
from codeintel.core.query_results import records_from_arrow_table
from codeintel.core.schemas.primitives import resolve_stable_sort_keys
from codeintel.core.schemas.service import get_schema_service

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetVerifyReport:
    """Report produced by dataset verification."""

    table_key: str
    manifest_path: Path
    missing_files: tuple[Path, ...]
    extra_files: tuple[Path, ...]

    @property
    def ok(self) -> bool:
        """Return True when no missing or extra files are detected."""
        return not self.missing_files and not self.extra_files


@dataclass(frozen=True, slots=True)
class DatasetVacuumReport:
    """Report produced by dataset vacuum/GC operations."""

    table_key: str
    manifest_path: Path
    removed_files: tuple[Path, ...]
    remaining_orphans: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class DatasetRewriteRequest:
    """Inputs for rewriting dataset partitions."""

    dataset_root: Path
    table_key: str
    snapshot_id: str
    partition_columns: tuple[str, ...]
    output_snapshot_id: str | None = None
    existing_data_behavior: ExistingDataBehavior = "delete_matching"


@dataclass(frozen=True, slots=True)
class DatasetCompactRequest:
    """Inputs for compacting dataset files."""

    dataset_root: Path
    table_key: str
    snapshot_id: str
    output_snapshot_id: str | None = None
    max_rows_per_file: int | None = None
    existing_data_behavior: ExistingDataBehavior = "delete_matching"


def rewrite_dataset_partitions(
    request: DatasetRewriteRequest,
) -> ArrowDatasetManifest:
    """Rewrite a dataset snapshot with a new partition spec.

    Parameters
    ----------
    request
        Request describing the dataset rewrite.

    Returns
    -------
    ArrowDatasetManifest
        Manifest for the rewritten dataset snapshot.
    """
    source_manifest = _require_manifest(
        dataset_root=request.dataset_root,
        table_key=request.table_key,
        snapshot_id=request.snapshot_id,
    )
    target_snapshot_id = request.output_snapshot_id or request.snapshot_id
    table = _scan_dataset_table(
        dataset_root=request.dataset_root,
        table_key=request.table_key,
        snapshot_id=request.snapshot_id,
    )
    finalized = _finalize_table_for_maintenance(table_key=request.table_key, table=table)
    options = ArrowDatasetWriteOptions(
        partition_columns=request.partition_columns,
        existing_data_behavior=request.existing_data_behavior,
        schema_hash=source_manifest.schema_hash,
        manifest_extras=source_manifest.extras,
        stable_sort_keys=_stable_sort_keys_for_table(request.table_key),
    )
    return write_dataset(
        dataset_root=request.dataset_root,
        table_key=request.table_key,
        snapshot_id=target_snapshot_id,
        data=finalized,
        options=options,
    )


def compact_dataset_files(
    request: DatasetCompactRequest,
) -> ArrowDatasetManifest:
    """Compact dataset files by rewriting into larger Parquet chunks.

    Parameters
    ----------
    request
        Request describing the dataset compaction.

    Returns
    -------
    ArrowDatasetManifest
        Manifest for the compacted dataset snapshot.
    """
    source_manifest = _require_manifest(
        dataset_root=request.dataset_root,
        table_key=request.table_key,
        snapshot_id=request.snapshot_id,
    )
    target_snapshot_id = request.output_snapshot_id or request.snapshot_id
    table = _scan_dataset_table(
        dataset_root=request.dataset_root,
        table_key=request.table_key,
        snapshot_id=request.snapshot_id,
    )
    finalized = _finalize_table_for_maintenance(table_key=request.table_key, table=table)
    options = ArrowDatasetWriteOptions(
        partition_columns=source_manifest.partition_columns,
        existing_data_behavior=request.existing_data_behavior,
        schema_hash=source_manifest.schema_hash,
        manifest_extras=source_manifest.extras,
        stable_sort_keys=_stable_sort_keys_for_table(request.table_key),
        max_rows_per_file=request.max_rows_per_file,
    )
    return write_dataset(
        dataset_root=request.dataset_root,
        table_key=request.table_key,
        snapshot_id=target_snapshot_id,
        data=finalized,
        options=options,
    )


def verify_dataset_manifest(*, manifest_path: Path) -> DatasetVerifyReport:
    """Verify that manifest files match on-disk Parquet files.

    Returns
    -------
    DatasetVerifyReport
        Report describing missing and extra Parquet files.
    """
    manifest = read_dataset_manifest(manifest_path)
    dataset_dir = manifest_path.parent
    expected = _expected_files(manifest, dataset_dir=dataset_dir)
    actual = _parquet_files(dataset_dir)
    missing = tuple(path for path in expected if not path.exists())
    extra = tuple(sorted(actual.difference(expected)))
    return DatasetVerifyReport(
        table_key=manifest.table_key,
        manifest_path=manifest_path,
        missing_files=missing,
        extra_files=extra,
    )


def vacuum_dataset_manifest(
    *,
    manifest_path: Path,
    dry_run: bool = True,
) -> DatasetVacuumReport:
    """Remove orphaned Parquet files not referenced by the manifest.

    Returns
    -------
    DatasetVacuumReport
        Report describing removed and remaining orphaned files.
    """
    manifest = read_dataset_manifest(manifest_path)
    dataset_dir = manifest_path.parent
    expected = _expected_files(manifest, dataset_dir=dataset_dir)
    orphans = tuple(sorted(_parquet_files(dataset_dir).difference(expected)))
    removed: list[Path] = []
    remaining: list[Path] = []
    for path in orphans:
        if dry_run:
            remaining.append(path)
            continue
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            remaining.append(path)
    return DatasetVacuumReport(
        table_key=manifest.table_key,
        manifest_path=manifest_path,
        removed_files=tuple(removed),
        remaining_orphans=tuple(remaining),
    )


def _scan_dataset_table(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> pa.Table:
    options = ParquetScanOptions(
        implicit_ordering=True,
        require_sequenced_output=True,
        metrics_enabled=True,
        provenance_columns=DEFAULT_ARROW_PROVENANCE_COLUMNS,
    )
    reader, telemetry = scan_parquet_dataset_with_telemetry(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=options,
    )
    if telemetry is not None:
        LOG.debug("Maintenance scan telemetry: %s", telemetry.to_mapping())
    if reader is None:
        msg = f"Dataset scan failed for {table_key}@{snapshot_id}"
        raise FileNotFoundError(msg)
    table = reader_to_table(reader)
    return normalize_table_for_compute(table)


def _finalize_table_for_maintenance(*, table_key: str, table: pa.Table) -> pa.Table:
    result = finalize_table(
        table,
        spec=FinalizeSpec(
            table_key=table_key,
            mode="tolerant",
            required_non_null=_required_non_null_columns(table_key),
            dedupe=FinalizeDedupe(enabled=False),
            context_fields=DEFAULT_ARROW_PROVENANCE_COLUMNS,
            emit_artifacts=True,
        ),
    )
    _log_finalize_warnings(table_key, result)
    return result.good


def _required_non_null_columns(table_key: str) -> tuple[str, ...]:
    try:
        schema = get_schema_service().get_table_schema(table_key)
    except RuntimeError:
        return ()
    if schema is None:
        return ()
    return tuple(column.name for column in schema.columns if not column.nullable)


def _stable_sort_keys_for_table(table_key: str) -> tuple[str, ...] | None:
    try:
        schema = get_schema_service().get_table_schema(table_key)
    except RuntimeError:
        return None
    return resolve_stable_sort_keys(schema)


def _log_finalize_warnings(table_key: str, result: FinalizeResult) -> None:
    if result.stats.num_rows:
        for row in records_from_arrow_table(result.stats):
            code = row.get("error_code")
            count = row.get("count")
            if isinstance(code, str):
                LOG.warning(
                    "Maintenance finalize error for %s: %s (%s rows)",
                    table_key,
                    code,
                    count,
                )
            else:
                LOG.warning("Maintenance finalize error for %s: %s", table_key, row)
    if result.alignment.num_rows:
        records = records_from_arrow_table(result.alignment)
        if records:
            row = records[0]
            LOG.warning(
                "Maintenance finalize alignment for %s: missing=%s extra=%s coerced=%s",
                table_key,
                row.get("missing_columns"),
                row.get("extra_columns"),
                row.get("coerced_columns"),
            )


def _require_manifest(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> ArrowDatasetManifest:
    path = dataset_manifest_path(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    if not path.is_file():
        msg = f"Dataset manifest not found: {path}"
        raise FileNotFoundError(msg)
    return read_dataset_manifest(path)


def _expected_files(
    manifest: ArrowDatasetManifest,
    *,
    dataset_dir: Path,
) -> set[Path]:
    if not manifest.files:
        return _parquet_files(dataset_dir)
    resolved: set[Path] = set()
    for rel_path in manifest.files:
        resolved.add((dataset_dir / rel_path).resolve())
    return resolved


def _parquet_files(dataset_dir: Path) -> set[Path]:
    return {path.resolve() for path in dataset_dir.rglob("*.parquet")}


__all__ = [
    "DatasetCompactRequest",
    "DatasetRewriteRequest",
    "DatasetVacuumReport",
    "DatasetVerifyReport",
    "compact_dataset_files",
    "rewrite_dataset_partitions",
    "vacuum_dataset_manifest",
    "verify_dataset_manifest",
]
