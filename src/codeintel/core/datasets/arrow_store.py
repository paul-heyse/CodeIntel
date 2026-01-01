"""Arrow dataset store for snapshot-scoped tables."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from codeintel.core.columnar.schema_metadata import merge_metadata
from codeintel.core.datasets.manifests import (
    dataset_manifest_path,
    read_dataset_manifest,
    write_dataset_manifest,
)
from codeintel.core.datasets.paths import dataset_snapshot_dir
from codeintel.core.datasets.scanning import (
    DatasetScanOptions,
    build_scanner,
    dataset_for_manifest,
)
from codeintel.core.manifests import ArrowDatasetManifest
from codeintel.core.schemas.arrow_metadata import arrow_schema_hash

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyarrow import RecordBatchReader, Table
    from pyarrow.dataset import FileWriteOptions

    type ArrowDatasetInput = Table | RecordBatchReader
else:
    type ArrowDatasetInput = object
ExistingDataBehavior = Literal["delete_matching", "error", "overwrite_or_ignore"]

LOG = logging.getLogger(__name__)


@runtime_checkable
class _SupportsAsPy(Protocol):
    def as_py(self) -> object: ...


@runtime_checkable
class _SupportsItem(Protocol):
    def item(self) -> object: ...


@runtime_checkable
class _SupportsRichComparison(Protocol):
    def __lt__(self, other: object) -> bool: ...

    def __gt__(self, other: object) -> bool: ...


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
    row_group_stats: Mapping[str, Mapping[str, float]] | None = None
    dictionary_encoding: Mapping[str, object] | None = None

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
        if self.row_group_stats:
            stats["row_group_stats"] = {
                key: dict(values) for key, values in self.row_group_stats.items()
            }
        if self.dictionary_encoding:
            stats["dictionary_encoding"] = dict(self.dictionary_encoding)
        return stats or None


@dataclass(frozen=True, slots=True)
class ArrowDatasetWriteOptions:
    """Options for writing Arrow dataset snapshots."""

    partition_columns: tuple[str, ...] = ()
    existing_data_behavior: ExistingDataBehavior = "delete_matching"
    persist_manifest: bool = True
    schema_hash: str | None = None
    manifest_extras: Mapping[str, object] | None = None
    schema_metadata: Mapping[str, object] | None = None
    max_rows_per_file: int | None = None
    row_group_size: int | None = None
    data_page_size: int | None = None
    compression: str | None = None
    dictionary_encode: bool = False
    dictionary_max_cardinality: int | None = None
    dictionary_encode_columns: tuple[str, ...] | None = None
    unify_dictionaries: bool = False


ArrowDatasetScanOptions = DatasetScanOptions


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
    start = perf_counter()
    resolved = options or ArrowDatasetWriteOptions()
    prepared = _apply_dictionary_options(data, resolved)
    prepared = _apply_schema_metadata(prepared, resolved.schema_metadata)
    snapshot_dir = dataset_snapshot_dir(
        dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    partitioning = _partitioning(resolved.partition_columns, schema=prepared.schema)
    parquet_format, file_options = _parquet_write_options(resolved)
    ds.write_dataset(
        prepared,
        str(snapshot_dir),
        format=parquet_format,
        partitioning=partitioning,
        existing_data_behavior=resolved.existing_data_behavior,
        file_options=file_options,
        max_rows_per_file=resolved.max_rows_per_file,
        max_rows_per_group=resolved.row_group_size,
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
    if LOG.isEnabledFor(logging.INFO):
        duration_ms = (perf_counter() - start) * 1000
        LOG.info(
            "Arrow dataset write: table=%s rows=%s files=%s duration_ms=%.2f",
            table_key,
            manifest.row_count,
            len(manifest.files),
            duration_ms,
        )
    return manifest


def scan_dataset(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> ds.Dataset:
    """Return a dataset handle for a snapshot.

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
    manifest_path = dataset_manifest_path(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    if manifest_path.is_file():
        try:
            manifest = read_dataset_manifest(manifest_path)
            return dataset_for_manifest(manifest=manifest, manifest_path=manifest_path)
        except (OSError, ValueError, pa.ArrowInvalid):
            LOG.debug("Falling back to raw dataset scan for %s", manifest_path)
    return ds.dataset(str(snapshot_dir), format="parquet")


def scan_dataset_scanner(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    options: ArrowDatasetScanOptions,
) -> ds.Scanner:
    """Return a dataset scanner for streaming reads.

    Parameters
    ----------
    dataset_root
        Root directory where Arrow datasets are stored.
    table_key
        Fully qualified table key (schema.table).
    snapshot_id
        Snapshot identifier used to scope the dataset.
    options
        Batch size and filter options for the scanner.

    Returns
    -------
    pyarrow.dataset.Scanner
        Scanner configured for streaming reads.
    """
    dataset = scan_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    return build_scanner(dataset, options=options)


def scan_dataset_reader(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    options: ArrowDatasetScanOptions,
) -> pa.RecordBatchReader:
    """Return a RecordBatchReader for a dataset snapshot.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader streaming record batches from the dataset.
    """
    scanner = scan_dataset_scanner(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=options,
    )
    return scanner.to_reader()


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
    parquet_rows = parquet_stats.rows_from_metadata if parquet_stats else None
    sort_keys = parquet_stats.sort_keys if parquet_stats and parquet_stats.sort_keys else None
    column_min_max = (
        parquet_stats.column_min_max if parquet_stats and parquet_stats.column_min_max else None
    )
    row_group_stats = (
        parquet_stats.row_group_stats if parquet_stats and parquet_stats.row_group_stats else None
    )
    dictionary_encoding = (
        parquet_stats.dictionary_encoding
        if parquet_stats and parquet_stats.dictionary_encoding
        else None
    )
    return ArrowDatasetStats(
        row_count=_count_rows(dataset, parquet_rows=parquet_rows),
        row_group_count=parquet_stats.row_group_count if parquet_stats else None,
        file_count=parquet_stats.file_count if parquet_stats else None,
        rows_from_metadata=parquet_stats.rows_from_metadata if parquet_stats else None,
        total_bytes=parquet_stats.total_bytes if parquet_stats else None,
        sort_keys=sort_keys,
        column_min_max=column_min_max,
        row_group_stats=row_group_stats,
        dictionary_encoding=dictionary_encoding,
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
    extras = dict(request.extras) if request.extras else {}
    parquet_extras = _parquet_extras(stats)
    if parquet_extras:
        existing = extras.get("parquet_stats")
        if isinstance(existing, Mapping):
            merged = dict(existing)
            merged.update(parquet_extras)
            extras["parquet_stats"] = merged
        else:
            extras["parquet_stats"] = parquet_extras
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
        extras=extras or None,
    )


def _parquet_extras(stats: ArrowDatasetStats) -> dict[str, object] | None:
    payload: dict[str, object] = {}
    if stats.row_group_stats:
        payload["row_group_stats"] = {
            key: dict(values) for key, values in stats.row_group_stats.items()
        }
    if stats.dictionary_encoding:
        payload["dictionary_encoding"] = dict(stats.dictionary_encoding)
    return payload or None


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


def _parquet_write_options(
    options: ArrowDatasetWriteOptions,
) -> tuple[ds.FileFormat, FileWriteOptions | None]:
    parquet_format = ds.ParquetFileFormat()
    make_options = getattr(parquet_format, "make_write_options", None)
    if not callable(make_options):
        return parquet_format, None
    signature = inspect.signature(make_options)
    kwargs: dict[str, object] = {}
    if options.compression and "compression" in signature.parameters:
        kwargs["compression"] = options.compression
    if options.data_page_size and "data_page_size" in signature.parameters:
        kwargs["data_page_size"] = options.data_page_size
    if (options.dictionary_encode or options.dictionary_encode_columns) and "use_dictionary" in (
        signature.parameters
    ):
        kwargs["use_dictionary"] = True
    if kwargs:
        return parquet_format, cast("FileWriteOptions", make_options(**kwargs))
    return parquet_format, cast("FileWriteOptions", make_options())


def _apply_dictionary_options(
    data: ArrowDatasetInput,
    options: ArrowDatasetWriteOptions,
) -> ArrowDatasetInput:
    if (
        not options.dictionary_encode
        and not options.dictionary_encode_columns
        and not (options.unify_dictionaries)
    ):
        return data
    encode_columns = (
        set(options.dictionary_encode_columns) if options.dictionary_encode_columns else None
    )
    encode_enabled = options.dictionary_encode or encode_columns is not None
    if isinstance(data, pa.Table):
        table = data
        if encode_enabled:
            table = _dictionary_encode_table(
                table,
                max_cardinality=options.dictionary_max_cardinality,
                encode_columns=encode_columns,
            )
        if options.unify_dictionaries:
            table = _unify_dictionaries(table)
        return table
    if isinstance(data, pa.RecordBatchReader) and options.dictionary_encode:
        LOG.debug("Dictionary encode skipped for stream input")
    if options.unify_dictionaries:
        LOG.debug("Dictionary unify skipped for stream input")
    return data


def _apply_schema_metadata(
    data: ArrowDatasetInput,
    metadata: Mapping[str, object] | None,
) -> ArrowDatasetInput:
    if not metadata:
        return data
    if isinstance(data, pa.Table):
        merged = merge_metadata(data.schema.metadata, metadata, overwrite=True)
        return data.replace_schema_metadata(merged)
    if isinstance(data, pa.RecordBatchReader):
        merged = merge_metadata(data.schema.metadata, metadata, overwrite=True)
        if merged == data.schema.metadata:
            return data
        schema = data.schema.with_metadata(merged)
        return pa.RecordBatchReader.from_batches(schema, data)
    return data


def _dictionary_encode_table(
    table: pa.Table,
    *,
    max_cardinality: int | None,
    encode_columns: set[str] | None,
) -> pa.Table:
    if max_cardinality is None or max_cardinality <= 0:
        return table
    arrays: list[pa.Array | pa.ChunkedArray] = []
    fields: list[pa.Field] = []
    for name in table.schema.names:
        column = table.column(name)
        if encode_columns is not None and name not in encode_columns:
            encoded = column
        else:
            encoded = _maybe_dictionary_encode_array(column, max_cardinality=max_cardinality)
        arrays.append(encoded)
        fields.append(pa.field(name, encoded.type))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _maybe_dictionary_encode_array(
    array: pa.Array | pa.ChunkedArray,
    *,
    max_cardinality: int,
) -> pa.Array | pa.ChunkedArray:
    data_type = array.type
    if not (pa.types.is_string(data_type) or pa.types.is_large_string(data_type)):
        return array
    distinct = _count_distinct(array)
    if distinct is None or distinct > max_cardinality:
        return array
    return _dictionary_encode(array)


def _count_distinct(array: pa.Array | pa.ChunkedArray) -> int | None:
    func = getattr(pc, "count_distinct", None)
    if not callable(func):
        return None
    result = func(array)
    return _coerce_int(_normalize_stat_value(result))


def _dictionary_encode(array: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    func = getattr(pc, "dictionary_encode", None)
    if not callable(func):
        return array
    return func(array)


def _unify_dictionaries(table: pa.Table) -> pa.Table:
    unify = getattr(table, "unify_dictionaries", None)
    if not callable(unify):
        return table
    try:
        return unify()
    except pa.ArrowInvalid:
        return table


def _count_rows(dataset: ds.Dataset, *, parquet_rows: int | None) -> int | None:
    if parquet_rows is not None:
        return parquet_rows
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
    return None


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
    row_group_stats: dict[str, dict[str, float]] | None
    dictionary_encoding: dict[str, object] | None


_DICTIONARY_ENCODINGS = frozenset({"PLAIN_DICTIONARY", "RLE_DICTIONARY"})


def _parquet_stats(files: tuple[str, ...]) -> _ParquetStats | None:
    if not files:
        return None
    row_groups = 0
    rows = 0
    total_bytes = 0
    sort_keys: list[str] = []
    sort_key_seen: set[str] = set()
    min_max: dict[str, tuple[object, object]] = {}
    row_group_rows: list[int] = []
    row_group_bytes: list[int] = []
    dictionary_counts: dict[str, int] = {}
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
            _extend_row_group_stats(
                metadata,
                row_group_rows=row_group_rows,
                row_group_bytes=row_group_bytes,
                dictionary_counts=dictionary_counts,
            )
    row_group_stats = _row_group_stats(row_group_rows, row_group_bytes)
    dictionary_encoding = _dictionary_encoding(dictionary_counts, row_group_count=row_groups)
    return _ParquetStats(
        row_group_count=row_groups,
        file_count=len(files),
        rows_from_metadata=rows,
        total_bytes=total_bytes,
        sort_keys=tuple(sort_keys),
        column_min_max=_min_max_to_mapping(min_max),
        row_group_stats=row_group_stats,
        dictionary_encoding=dictionary_encoding,
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


def _extend_row_group_stats(
    metadata: pq.FileMetaData,
    *,
    row_group_rows: list[int],
    row_group_bytes: list[int],
    dictionary_counts: dict[str, int],
) -> None:
    for group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(group_index)
        row_count = _coerce_int(row_group.num_rows)
        if row_count is not None:
            row_group_rows.append(row_count)
        byte_size = _coerce_int(row_group.total_byte_size)
        if byte_size is not None:
            row_group_bytes.append(byte_size)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            if not _has_dictionary_encoding(column):
                continue
            column_name = column.path_in_schema
            dictionary_counts[column_name] = dictionary_counts.get(column_name, 0) + 1


def _has_dictionary_encoding(column: pq.ColumnChunkMetaData) -> bool:
    encodings = getattr(column, "encodings", None)
    if not encodings:
        return False
    return any(str(encoding).upper() in _DICTIONARY_ENCODINGS for encoding in encodings)


def _row_group_stats(
    row_group_rows: Sequence[int],
    row_group_bytes: Sequence[int],
) -> dict[str, dict[str, float]] | None:
    stats: dict[str, dict[str, float]] = {}
    row_stats = _summary_stats(row_group_rows)
    if row_stats is not None:
        stats["rows"] = row_stats
    byte_stats = _summary_stats(row_group_bytes)
    if byte_stats is not None:
        stats["bytes"] = byte_stats
    return stats or None


def _summary_stats(values: Sequence[int]) -> dict[str, float] | None:
    if not values:
        return None
    count = len(values)
    total = sum(values)
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "avg": float(total / count),
    }


def _dictionary_encoding(
    dictionary_counts: Mapping[str, int],
    *,
    row_group_count: int,
) -> dict[str, object] | None:
    if not dictionary_counts:
        return None
    return {
        "row_groups": row_group_count,
        "columns": dict(sorted(dictionary_counts.items())),
    }


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
    if isinstance(value, _SupportsAsPy):
        return value.as_py()
    if isinstance(value, _SupportsItem):
        try:
            return value.item()
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
    if isinstance(candidate, _SupportsRichComparison) and isinstance(
        current, _SupportsRichComparison
    ):
        try:
            return candidate if candidate < current else current
        except TypeError:
            return current
    return current


def _safe_max(current: object, candidate: object) -> object:
    if isinstance(candidate, _SupportsRichComparison) and isinstance(
        current, _SupportsRichComparison
    ):
        try:
            return candidate if candidate > current else current
        except TypeError:
            return current
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
        result: object = value
    elif isinstance(value, bytes):
        result = value.hex()
    elif isinstance(value, (datetime, date)):
        result = value.isoformat()
    elif isinstance(value, Decimal):
        result = str(value)
    elif isinstance(value, _SupportsAsPy):
        result = _json_safe_value(value.as_py())
    elif isinstance(value, _SupportsItem):
        try:
            result = _json_safe_value(value.item())
        except (TypeError, ValueError, OverflowError):
            result = str(value)
    else:
        result = str(value)
    return result


__all__ = [
    "ArrowDatasetInput",
    "ArrowDatasetManifestRequest",
    "ArrowDatasetScanOptions",
    "ArrowDatasetStats",
    "ArrowDatasetWriteOptions",
    "ExistingDataBehavior",
    "build_dataset_manifest",
    "dataset_stats",
    "scan_dataset",
    "scan_dataset_reader",
    "scan_dataset_scanner",
    "write_dataset",
]
