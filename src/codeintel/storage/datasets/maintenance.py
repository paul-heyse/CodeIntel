"""Arrow dataset maintenance utilities (rewrite, compact, vacuum, verify)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from codeintel.core.columnar.arrowdsl import ExecutionContext, ExecutionPlan, run_pipeline
from codeintel.core.columnar.conversion import record_batch_reader_from_iterable
from codeintel.core.columnar.dedupe_ops import DedupeTier
from codeintel.core.columnar.finalize_ops import (
    FinalizeDedupe,
    FinalizeResult,
    finalize_spec_for_table,
)
from codeintel.core.columnar.kernels import SortKey
from codeintel.core.columnar.readers import empty_reader_from_schema
from codeintel.core.constants import DEFAULT_ARROW_PROVENANCE_COLUMNS, DEFAULT_ARROW_USE_THREADS
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
    reader = _scan_dataset_reader(
        dataset_root=request.dataset_root,
        table_key=request.table_key,
        snapshot_id=request.snapshot_id,
    )
    finalized = _finalize_reader_for_maintenance(
        table_key=request.table_key,
        reader=reader,
    )
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
    reader = _scan_dataset_reader(
        dataset_root=request.dataset_root,
        table_key=request.table_key,
        snapshot_id=request.snapshot_id,
    )
    finalized = _finalize_reader_for_maintenance(
        table_key=request.table_key,
        reader=reader,
    )
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


def _scan_dataset_reader(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> pa.RecordBatchReader:
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
    return reader


def _finalize_reader_for_maintenance(
    *,
    table_key: str,
    reader: pa.RecordBatchReader,
) -> pa.RecordBatchReader:
    stable_sort_keys = _stable_sort_keys_for_table(table_key)
    execution_ctx = _execution_context_for_maintenance(
        table_key=table_key,
        stable_sort_keys=stable_sort_keys,
    )
    order_by = _order_by_for_table(
        table_key,
        stable_sort_keys=stable_sort_keys,
        determinism=execution_ctx.determinism,
    )
    dedupe_keys = _dedupe_keys_for_table(table_key)
    finalize_spec = finalize_spec_for_table(
        table_key,
        mode="tolerant",
        dedupe=FinalizeDedupe(
            enabled=False,
            keys=dedupe_keys,
            tie_breakers=order_by,
            tier=execution_ctx.determinism,
            strategy="order_independent",
        ),
        context_fields=DEFAULT_ARROW_PROVENANCE_COLUMNS,
        emit_artifacts=True,
        order_by=order_by,
    )

    def _iter_batches() -> Iterable[pa.RecordBatch]:
        for batch in reader:
            if batch.num_rows == 0:
                continue
            table = pa.Table.from_batches([batch], schema=batch.schema)
            result = run_pipeline(
                plan=ExecutionPlan.from_table(table),
                finalize=finalize_spec,
                ctx=execution_ctx,
            )
            _log_finalize_warnings(table_key, result)
            yield from result.good.to_batches(max_chunksize=batch.num_rows)

    finalized = record_batch_reader_from_iterable(_iter_batches(), empty_policy="none")
    if finalized is None:
        return empty_reader_from_schema(reader.schema)
    return finalized


def _stable_sort_keys_for_table(table_key: str) -> tuple[str, ...] | None:
    try:
        schema = get_schema_service().get_table_schema(table_key)
    except RuntimeError:
        return None
    return resolve_stable_sort_keys(schema)


def _dedupe_keys_for_table(table_key: str) -> tuple[str, ...]:
    try:
        schema = get_schema_service().get_table_schema(table_key)
    except RuntimeError:
        return ()
    if schema is None or not schema.primary_key:
        return ()
    return tuple(schema.primary_key)


def _resolve_combine_chunks_for_table(table_key: str) -> bool:
    try:
        schema = get_schema_service().get_table_schema(table_key)
    except RuntimeError:
        return True
    if schema is None:
        return True
    policy = schema.write_policy
    if policy is not None and policy.combine_chunks is not None:
        return policy.combine_chunks
    return True


def _execution_context_for_maintenance(
    *,
    table_key: str,
    stable_sort_keys: tuple[str, ...] | None,
) -> ExecutionContext:
    determinism = "throughput" if stable_sort_keys == () else "canonical"
    return ExecutionContext(
        use_threads=DEFAULT_ARROW_USE_THREADS,
        determinism=determinism,
        combine_chunks=_resolve_combine_chunks_for_table(table_key),
    )


def _order_by_for_table(
    table_key: str,
    *,
    stable_sort_keys: tuple[str, ...] | None,
    determinism: DedupeTier,
) -> tuple[SortKey, ...]:
    if stable_sort_keys:
        return tuple((key, "ascending") for key in stable_sort_keys)
    if determinism == "canonical":
        msg = f"Maintenance finalize requires order_by keys for canonical determinism: {table_key}"
        raise ValueError(msg)
    return ()


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
