"""Arrow dataset store for snapshot-scoped tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from codeintel.core.manifests import ArrowDatasetManifest
from codeintel.storage.datasets.manifests import dataset_manifest_path, write_dataset_manifest
from codeintel.storage.datasets.paths import dataset_snapshot_dir
from codeintel.storage.schema import arrow_schema_hash

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from pyarrow import RecordBatchReader, Table

    type ArrowDatasetInput = Table | RecordBatchReader
else:
    type ArrowDatasetInput = object
ExistingDataBehavior = Literal["delete_matching", "error", "overwrite_or_ignore"]


@dataclass(frozen=True, slots=True)
class ArrowDatasetStats:
    """Summary statistics for an Arrow dataset snapshot."""

    row_count: int | None
    row_group_count: int | None = None
    file_count: int | None = None
    rows_from_metadata: int | None = None
    total_bytes: int | None = None
    sort_keys: tuple[str, ...] | None = None
    column_min_max: Mapping[str, Mapping[str, object]] | None = None

    def to_mapping(self) -> dict[str, object] | None:
        """Return a stats mapping suitable for manifest storage.

        Returns
        -------
        dict[str, object] | None
            Manifest-ready stats mapping or None when no stats are present.
        """
        stats: dict[str, object] = {}
        if self.row_group_count is not None:
            stats["row_groups"] = self.row_group_count
        if self.file_count is not None:
            stats["file_count"] = self.file_count
        if self.rows_from_metadata is not None:
            stats["rows_from_metadata"] = self.rows_from_metadata
        if self.total_bytes is not None:
            stats["total_bytes"] = self.total_bytes
        if self.sort_keys:
            stats["sort_keys"] = list(self.sort_keys)
        if self.column_min_max:
            stats["min_max"] = {
                column: dict(values) for column, values in self.column_min_max.items()
            }
        return stats or None


@dataclass(frozen=True, slots=True)
class ArrowDatasetWriteOptions:
    """Options for writing Arrow dataset snapshots."""

    partition_columns: tuple[str, ...] = ()
    existing_data_behavior: ExistingDataBehavior = "delete_matching"
    persist_manifest: bool = True
    schema_hash: str | None = None
    manifest_extras: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ArrowDatasetManifestRequest:
    """Arguments for building Arrow dataset manifests."""

    table_key: str
    snapshot_id: str
    partition_columns: tuple[str, ...]
    schema_hash: str | None = None
    extras: Mapping[str, object] | None = None
    created_at: str | None = None


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
    partitioning = _partitioning(resolved.partition_columns, schema=data.schema)
    ds.write_dataset(
        data,
        str(snapshot_dir),
        format="parquet",
        partitioning=partitioning,
        existing_data_behavior=resolved.existing_data_behavior,
    )
    dataset = ds.dataset(str(snapshot_dir), format="parquet")
    request = ArrowDatasetManifestRequest(
        table_key=table_key,
        snapshot_id=snapshot_id,
        partition_columns=resolved.partition_columns,
        schema_hash=resolved.schema_hash,
        extras=resolved.manifest_extras,
    )
    manifest = build_dataset_manifest(
        dataset=dataset,
        snapshot_dir=snapshot_dir,
        request=request,
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
    files = tuple(dataset.files)
    parquet_stats = _parquet_stats(files)
    sort_keys = parquet_stats.sort_keys if parquet_stats and parquet_stats.sort_keys else None
    column_min_max = (
        parquet_stats.column_min_max if parquet_stats and parquet_stats.column_min_max else None
    )
    return ArrowDatasetStats(
        row_count=_count_rows(dataset),
        row_group_count=parquet_stats.row_group_count if parquet_stats else None,
        file_count=parquet_stats.file_count if parquet_stats else None,
        rows_from_metadata=parquet_stats.rows_from_metadata if parquet_stats else None,
        total_bytes=parquet_stats.total_bytes if parquet_stats else None,
        sort_keys=sort_keys,
        column_min_max=column_min_max,
    )


def build_dataset_manifest(
    *,
    dataset: ds.Dataset,
    snapshot_dir: Path,
    request: ArrowDatasetManifestRequest,
) -> ArrowDatasetManifest:
    """Return a dataset manifest for an on-disk dataset snapshot.

    Parameters
    ----------
    dataset
        Arrow dataset handle for the snapshot.
    snapshot_dir
        Snapshot directory containing the dataset files.
    request
        Manifest parameters for the snapshot.

    Returns
    -------
    ArrowDatasetManifest
        Manifest describing the on-disk dataset snapshot.
    """
    stats = dataset_stats(dataset)
    files = _relative_files(dataset.files, base_dir=snapshot_dir)
    resolved_hash = request.schema_hash or arrow_schema_hash(dataset.schema)
    return ArrowDatasetManifest(
        dataset_id=request.table_key,
        snapshot_id=request.snapshot_id,
        table_key=request.table_key,
        schema_hash=resolved_hash,
        partition_columns=tuple(str(column) for column in request.partition_columns),
        files=files,
        row_count=stats.row_count,
        stats=stats.to_mapping(),
        created_at=request.created_at or datetime.now(tz=UTC).isoformat(),
        extras=dict(request.extras) if request.extras else None,
    )


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


@dataclass(frozen=True, slots=True)
class _ParquetStats:
    row_group_count: int
    file_count: int
    rows_from_metadata: int
    total_bytes: int
    sort_keys: tuple[str, ...]
    column_min_max: dict[str, dict[str, object]]


def _parquet_stats(files: tuple[str, ...]) -> _ParquetStats | None:
    if not files:
        return None
    row_groups = 0
    rows = 0
    total_bytes = 0
    sort_keys: list[str] = []
    sort_key_seen: set[str] = set()
    min_max: dict[str, tuple[object, object]] = {}
    for path in files:
        file_path = Path(path)
        try:
            total_bytes += file_path.stat().st_size
        except OSError:
            continue
        try:
            parquet_file = pq.ParquetFile(file_path)
        except (OSError, pa.ArrowInvalid):
            continue
        row_groups += parquet_file.num_row_groups
        metadata = parquet_file.metadata
        if metadata is not None:
            rows += metadata.num_rows
            _extend_sort_keys(
                sort_keys,
                sort_key_seen,
                metadata=metadata,
                schema=parquet_file.schema_arrow,
            )
            _merge_min_max(metadata, min_max)
    return _ParquetStats(
        row_group_count=row_groups,
        file_count=len(files),
        rows_from_metadata=rows,
        total_bytes=total_bytes,
        sort_keys=tuple(sort_keys),
        column_min_max=_min_max_to_mapping(min_max),
    )


def _extend_sort_keys(
    keys: list[str],
    seen: set[str],
    *,
    metadata: pq.FileMetaData,
    schema: pa.Schema,
) -> None:
    sort_columns = getattr(metadata, "sorting_columns", None)
    if not sort_columns:
        return
    names = schema.names
    for column in sort_columns:
        index = getattr(column, "column_idx", None)
        if index is None or index < 0 or index >= len(names):
            continue
        name = names[index]
        if name in seen:
            continue
        seen.add(name)
        keys.append(name)


def _merge_min_max(
    metadata: pq.FileMetaData,
    accumulator: dict[str, tuple[object, object]],
) -> None:
    for group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            stats = column.statistics
            if stats is None or not getattr(stats, "has_min_max", False):
                continue
            min_value, max_value = _extract_min_max(stats)
            if min_value is None or max_value is None:
                continue
            column_name = column.path_in_schema
            accumulator[column_name] = _merge_min_max_pair(
                accumulator.get(column_name),
                min_value=min_value,
                max_value=max_value,
            )


def _extract_min_max(stats: object) -> tuple[object | None, object | None]:
    min_value = _normalize_stat_value(getattr(stats, "min", None))
    max_value = _normalize_stat_value(getattr(stats, "max", None))
    if min_value is None or max_value is None:
        return None, None
    return min_value, max_value


def _normalize_stat_value(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, pa.Scalar):
        return value.as_py()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError, OverflowError):
            return value
    return value


def _merge_min_max_pair(
    current: tuple[object, object] | None,
    *,
    min_value: object,
    max_value: object,
) -> tuple[object, object]:
    if current is None:
        return min_value, max_value
    current_min, current_max = current
    return _safe_min(current_min, min_value), _safe_max(current_max, max_value)


def _safe_min(current: object, candidate: object) -> object:
    try:
        return candidate if candidate < current else current
    except TypeError:
        return current


def _safe_max(current: object, candidate: object) -> object:
    try:
        return candidate if candidate > current else current
    except TypeError:
        return current


def _min_max_to_mapping(
    min_max: dict[str, tuple[object, object]],
) -> dict[str, dict[str, object]]:
    return {
        column: {
            "min": _json_safe_value(values[0]),
            "max": _json_safe_value(values[1]),
        }
        for column, values in min_max.items()
    }


def _json_safe_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        return _json_safe_value(as_py())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe_value(item())
        except (TypeError, ValueError, OverflowError):
            return str(value)
    return str(value)


__all__ = [
    "ArrowDatasetInput",
    "ArrowDatasetManifestRequest",
    "ArrowDatasetStats",
    "ArrowDatasetWriteOptions",
    "ExistingDataBehavior",
    "build_dataset_manifest",
    "dataset_stats",
    "scan_dataset",
    "write_dataset",
]
