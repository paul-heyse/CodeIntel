"""Arrow dataset maintenance utilities (rewrite, compact, vacuum, verify)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.manifests import ArrowDatasetManifest
from codeintel.storage.datasets.arrow_store import (
    ArrowDatasetManifestRequest,
    ExistingDataBehavior,
    build_dataset_manifest,
)
from codeintel.storage.datasets.paths import dataset_snapshot_dir
from codeintel.storage.manifests import (
    dataset_manifest_path,
    read_dataset_manifest,
    write_dataset_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    source_dir = dataset_snapshot_dir(
        request.dataset_root,
        table_key=request.table_key,
        snapshot_id=request.snapshot_id,
    )
    target_dir = dataset_snapshot_dir(
        request.dataset_root,
        table_key=request.table_key,
        snapshot_id=target_snapshot_id,
    )
    dataset = ds.dataset(str(source_dir), format="parquet")
    ds.write_dataset(
        dataset,
        str(target_dir),
        format="parquet",
        partitioning=_partitioning(request.partition_columns, schema=dataset.schema),
        existing_data_behavior=request.existing_data_behavior,
    )
    return _finalize_manifest(
        dataset_root=request.dataset_root,
        table_key=request.table_key,
        snapshot_id=target_snapshot_id,
        partition_columns=request.partition_columns,
        source_manifest=source_manifest,
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
    source_dir = dataset_snapshot_dir(
        request.dataset_root,
        table_key=request.table_key,
        snapshot_id=request.snapshot_id,
    )
    target_dir = dataset_snapshot_dir(
        request.dataset_root,
        table_key=request.table_key,
        snapshot_id=target_snapshot_id,
    )
    dataset = ds.dataset(str(source_dir), format="parquet")
    ds.write_dataset(
        dataset,
        str(target_dir),
        format="parquet",
        partitioning=_partitioning(source_manifest.partition_columns, schema=dataset.schema),
        existing_data_behavior=request.existing_data_behavior,
        max_rows_per_file=request.max_rows_per_file,
    )
    return _finalize_manifest(
        dataset_root=request.dataset_root,
        table_key=request.table_key,
        snapshot_id=target_snapshot_id,
        partition_columns=source_manifest.partition_columns,
        source_manifest=source_manifest,
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


def _finalize_manifest(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    partition_columns: tuple[str, ...],
    source_manifest: ArrowDatasetManifest,
) -> ArrowDatasetManifest:
    snapshot_dir = dataset_snapshot_dir(
        dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    dataset = ds.dataset(str(snapshot_dir), format="parquet")
    request = ArrowDatasetManifestRequest(
        table_key=table_key,
        snapshot_id=snapshot_id,
        partition_columns=partition_columns,
        schema_hash=source_manifest.schema_hash,
        extras=source_manifest.extras,
    )
    manifest = build_dataset_manifest(
        dataset=dataset,
        snapshot_dir=snapshot_dir,
        request=request,
    )
    manifest_path = dataset_manifest_path(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    write_dataset_manifest(manifest_path, manifest)
    return manifest


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


def _partitioning(
    columns: Sequence[str],
    *,
    schema: pa.Schema | None,
) -> ds.Partitioning | None:
    if not columns:
        return None
    if schema is None:
        msg = "Partitioning requires a schema when partition columns are provided"
        raise ValueError(msg)
    try:
        fields = [schema.field(str(column)) for column in columns]
    except KeyError as exc:
        msg = f"Partition columns missing from schema: {columns}"
        raise ValueError(msg) from exc
    return ds.partitioning(schema=pa.schema(fields))


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
