"""Arrow dataset store for snapshot-scoped tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.manifests import ArrowDatasetManifest
from codeintel.storage.datasets.manifests import dataset_manifest_path, write_dataset_manifest
from codeintel.storage.datasets.paths import dataset_snapshot_dir
from codeintel.storage.schema import arrow_schema_hash

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pyarrow import RecordBatchReader, Table

    type ArrowDatasetInput = Table | RecordBatchReader
else:
    type ArrowDatasetInput = object
ExistingDataBehavior = Literal["delete_matching", "error", "overwrite_or_ignore"]


@dataclass(frozen=True, slots=True)
class ArrowDatasetStats:
    """Summary statistics for an Arrow dataset snapshot."""

    row_count: int | None


@dataclass(frozen=True, slots=True)
class ArrowDatasetWriteOptions:
    """Options for writing Arrow dataset snapshots."""

    partition_columns: tuple[str, ...] = ()
    existing_data_behavior: ExistingDataBehavior = "delete_matching"
    persist_manifest: bool = True


def write_dataset(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    data: ArrowDatasetInput,
    options: ArrowDatasetWriteOptions | None = None,
) -> ArrowDatasetManifest:
    """Write a dataset snapshot and return its manifest.

    Parameters
    ----------
    dataset_root
        Root directory where Arrow datasets are stored.
    table_key
        Fully qualified table key (schema.table).
    snapshot_id
        Snapshot identifier used to scope the dataset.
    data
        Tabular Arrow data to write.
    options
        Optional write options for partitioning and manifest persistence.

    Returns
    -------
    ArrowDatasetManifest
        Manifest describing the written dataset snapshot.
    """
    resolved = options or ArrowDatasetWriteOptions()
    snapshot_dir = dataset_snapshot_dir(
        dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    partitioning = _partitioning(resolved.partition_columns)
    ds.write_dataset(
        data,
        str(snapshot_dir),
        format="parquet",
        partitioning=partitioning,
        existing_data_behavior=resolved.existing_data_behavior,
    )
    dataset = ds.dataset(str(snapshot_dir), format="parquet")
    stats = dataset_stats(dataset)
    files = _relative_files(dataset.files, base_dir=snapshot_dir)
    manifest = ArrowDatasetManifest(
        dataset_id=table_key,
        snapshot_id=snapshot_id,
        table_key=table_key,
        schema_hash=arrow_schema_hash(dataset.schema),
        partition_columns=resolved.partition_columns,
        files=files,
        row_count=stats.row_count,
        created_at=datetime.now(tz=UTC).isoformat(),
    )
    if resolved.persist_manifest:
        manifest_path = dataset_manifest_path(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
        write_dataset_manifest(manifest_path, manifest)
    return manifest


def scan_dataset(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> ds.Dataset:
    """Return a dataset scanner for a snapshot.

    Parameters
    ----------
    dataset_root
        Root directory where Arrow datasets are stored.
    table_key
        Fully qualified table key (schema.table).
    snapshot_id
        Snapshot identifier used to scope the dataset.

    Returns
    -------
    pyarrow.dataset.Dataset
        Dataset handle for scanning.

    Raises
    ------
    FileNotFoundError
        If the dataset snapshot directory does not exist.
    """
    snapshot_dir = dataset_snapshot_dir(
        dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    if not snapshot_dir.is_dir():
        msg = f"Dataset snapshot not found: {snapshot_dir}"
        raise FileNotFoundError(msg)
    return ds.dataset(str(snapshot_dir), format="parquet")


def dataset_stats(dataset: ds.Dataset) -> ArrowDatasetStats:
    """Return lightweight dataset statistics.

    Parameters
    ----------
    dataset
        Arrow dataset handle.

    Returns
    -------
    ArrowDatasetStats
        Statistics derived from the dataset.
    """
    return ArrowDatasetStats(row_count=_count_rows(dataset))


def _partitioning(columns: Sequence[str]) -> ds.Partitioning | None:
    if not columns:
        return None
    return ds.partitioning([str(column) for column in columns])


def _count_rows(dataset: ds.Dataset) -> int | None:
    counter = getattr(dataset, "count_rows", None)
    if callable(counter):
        try:
            coerced = _coerce_int(counter())
            if coerced is not None:
                return coerced
        except (TypeError, ValueError, pa.ArrowInvalid):
            pass
    scanner = dataset.scanner()
    scanner_counter = getattr(scanner, "count_rows", None)
    if callable(scanner_counter):
        try:
            coerced = _coerce_int(scanner_counter())
            if coerced is not None:
                return coerced
        except (TypeError, ValueError, pa.ArrowInvalid):
            pass
    table = scanner.to_table()
    return table.num_rows


def _relative_files(files: Iterable[str], *, base_dir: Path) -> tuple[str, ...]:
    base_dir = base_dir.resolve()
    normalized: list[str] = []
    for file in files:
        path = Path(file)
        try:
            normalized.append(str(path.resolve().relative_to(base_dir)))
        except (ValueError, RuntimeError):
            normalized.append(str(path))
    return tuple(normalized)


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


__all__ = [
    "ArrowDatasetInput",
    "ArrowDatasetStats",
    "ArrowDatasetWriteOptions",
    "ExistingDataBehavior",
    "dataset_stats",
    "scan_dataset",
    "write_dataset",
]
