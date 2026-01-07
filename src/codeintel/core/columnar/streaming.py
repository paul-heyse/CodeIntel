"""Arrow-first streaming helpers for datasets and readers."""

from __future__ import annotations

import contextlib
import inspect
import os
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from codeintel.core.columnar.schema import DEFAULT_SCHEMA_PROMOTE_OPTIONS, SchemaPromoteOptions
from codeintel.core.columnar.schema_ops import unify_schemas
from codeintel.core.constants import (
    DEFAULT_ARROW_BATCH_READAHEAD,
    DEFAULT_ARROW_BATCH_SIZE,
    DEFAULT_ARROW_CPU_COUNT,
    DEFAULT_ARROW_FRAGMENT_READAHEAD,
    DEFAULT_ARROW_IO_THREAD_COUNT,
    DEFAULT_ARROW_IO_THREAD_MULTIPLIER,
    DEFAULT_ARROW_MIN_IO_THREADS,
    DEFAULT_ARROW_USE_THREADS,
)
from codeintel.core.manifests import ArrowDatasetManifest

if TYPE_CHECKING:
    from polars import LazyFrame
    from pyarrow.dataset import Scanner

    type PolarsLazyFrame = LazyFrame
else:
    type PolarsLazyFrame = object

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None


@dataclass(frozen=True, slots=True)
class DatasetScanOptions:
    """Options for Arrow dataset scanning."""

    batch_size: int
    batch_readahead: int | None = DEFAULT_ARROW_BATCH_READAHEAD
    fragment_readahead: int | None = DEFAULT_ARROW_FRAGMENT_READAHEAD
    filter_expression: ds.Expression | None = None
    use_threads: bool | None = DEFAULT_ARROW_USE_THREADS
    memory_pool: pa.MemoryPool | None = None
    schema: pa.Schema | None = None
    columns: Sequence[str] | None = None
    unify_schemas: bool = False
    schema_promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS
    metrics_enabled: bool = False


@dataclass(frozen=True, slots=True)
class QueryPlanSpec:
    """Shared query plan details for dataset scanning."""

    table_key: str
    columns: tuple[str, ...]
    filter_expression: ds.Expression | None


_METADATA_FILENAME = "_metadata"
_COMMON_METADATA_FILENAME = "_common_metadata"
_ARROW_THREADING_CONFIGURED = threading.Event()


def _resolve_arrow_cpu_count(default_count: int | None) -> int:
    if default_count is not None and default_count > 0:
        return default_count
    detected = os.cpu_count() or 1
    return max(1, detected)


def _resolve_arrow_io_thread_count(
    default_count: int | None,
    *,
    cpu_count: int,
) -> int:
    if default_count is not None and default_count > 0:
        return default_count
    scaled = cpu_count * DEFAULT_ARROW_IO_THREAD_MULTIPLIER
    return max(DEFAULT_ARROW_MIN_IO_THREADS, scaled)


def configure_arrow_threading(
    *,
    cpu_count: int | None = DEFAULT_ARROW_CPU_COUNT,
    io_thread_count: int | None = DEFAULT_ARROW_IO_THREAD_COUNT,
) -> None:
    """Apply Arrow threading defaults for compute and dataset scans."""
    if _ARROW_THREADING_CONFIGURED.is_set():
        return
    _ARROW_THREADING_CONFIGURED.set()
    resolved_cpu = _resolve_arrow_cpu_count(cpu_count)
    resolved_io = _resolve_arrow_io_thread_count(io_thread_count, cpu_count=resolved_cpu)
    set_cpu = getattr(pa, "set_cpu_count", None)
    if callable(set_cpu):
        with contextlib.suppress(TypeError, ValueError, pa.ArrowInvalid):
            set_cpu(resolved_cpu)
    set_io = getattr(pa, "set_io_thread_count", None)
    if callable(set_io):
        with contextlib.suppress(TypeError, ValueError, pa.ArrowInvalid):
            set_io(resolved_io)


def resolve_partitioning(
    *,
    manifest: ArrowDatasetManifest,
    schema: pa.Schema | None,
) -> ds.Partitioning | str | None:
    """Resolve dataset partitioning from manifest metadata.

    Parameters
    ----------
    manifest
        Manifest describing the dataset layout.
    schema
        Optional Arrow schema used to validate partition columns.

    Returns
    -------
    pyarrow.dataset.Partitioning | str | None
        Partitioning definition for Arrow dataset construction.
    """
    if not manifest.partition_columns:
        return None
    if schema is None:
        return "hive"
    if any(column not in schema.names for column in manifest.partition_columns):
        return "hive"
    fields = [schema.field(column) for column in manifest.partition_columns]
    return ds.partitioning(schema=pa.schema(fields))


def dataset_for_manifest(
    *,
    manifest: ArrowDatasetManifest,
    manifest_path: Path,
) -> ds.Dataset:
    """Return a PyArrow dataset for a manifest payload.

    Parameters
    ----------
    manifest
        Dataset manifest metadata.
    manifest_path
        Path to the manifest file on disk.

    Returns
    -------
    pyarrow.dataset.Dataset
        Dataset built from the manifest metadata.
    """
    dataset_dir = manifest_path.parent.resolve()
    metadata_path = _dataset_metadata_path(dataset_dir)
    common_metadata_path = _common_metadata_path(dataset_dir)
    schema = _schema_from_common_metadata(dataset_dir)
    parquet_format = _parquet_format_for_manifest(manifest.extras, schema=schema)
    partitioning = resolve_partitioning(manifest=manifest, schema=schema)
    metadata_source = metadata_path or common_metadata_path
    if metadata_source is not None:
        dataset = _dataset_from_metadata(
            metadata_source,
            dataset_dir=dataset_dir,
            partitioning=partitioning,
            schema=schema,
            parquet_format=parquet_format,
        )
        if dataset is not None:
            return dataset
    if manifest.files:
        paths = [str(dataset_dir / path) for path in manifest.files]
        return ds.dataset(paths, format=parquet_format, partitioning=partitioning, schema=schema)
    return ds.dataset(
        str(dataset_dir), format=parquet_format, partitioning=partitioning, schema=schema
    )


def build_scanner(dataset: ds.Dataset, *, options: DatasetScanOptions) -> Scanner:
    """Build a dataset scanner using shared scan options.

    Parameters
    ----------
    dataset
        Arrow dataset to scan.
    options
        Scan configuration options.

    Returns
    -------
    pyarrow.dataset.Scanner
        Scanner configured for the dataset.
    """
    configure_arrow_threading()
    schema = _resolve_scan_schema(dataset, options)
    scan_kwargs = _build_scan_kwargs(options, schema)
    filter_expression = options.filter_expression
    if filter_expression is None:
        return _scanner_with_schema(dataset, scan_kwargs)

    fragments = _fragments_for_filter(dataset, filter_expression)
    resolved_schema = schema or dataset.schema
    fragment_scanner = _scanner_from_fragments(
        fragments,
        resolved_schema,
        scan_kwargs,
        filter_expression=filter_expression,
    )
    if fragment_scanner is not None:
        return fragment_scanner

    scan_kwargs["filter"] = filter_expression
    return _scanner_with_schema(dataset, scan_kwargs)


def unify_dataset_schema(
    dataset: ds.Dataset,
    *,
    schema_promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS,
) -> pa.Schema | None:
    """Return a unified schema for a dataset when fragments diverge.

    Parameters
    ----------
    dataset
        Dataset to inspect for fragment schema differences.
    schema_promote_options
        Options used when promoting fragment schemas.

    Returns
    -------
    pyarrow.Schema | None
        Unified schema if one can be resolved.
    """
    fragments = _dataset_fragments(dataset)
    if fragments is None:
        return dataset.schema
    schemas: list[pa.Schema] = []
    for fragment in fragments:
        schema = getattr(fragment, "physical_schema", None)
        if isinstance(schema, pa.Schema):
            schemas.append(schema)
    if not schemas:
        return dataset.schema
    if len(schemas) == 1:
        return schemas[0]
    try:
        return unify_schemas(schemas, promote_options=schema_promote_options)
    except (TypeError, ValueError, pa.ArrowInvalid):
        return dataset.schema


def empty_reader_from_schema(schema: pa.Schema) -> pa.RecordBatchReader:
    """Return an empty reader with the provided schema.

    Parameters
    ----------
    schema
        Schema to associate with the empty reader.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader with no batches and the provided schema.
    """
    return pa.RecordBatchReader.from_batches(schema, [])


def scan_dataset_reader(
    dataset_dir: Path,
    *,
    options: DatasetScanOptions | None = None,
) -> pa.RecordBatchReader | None:
    """Return a streaming reader for a dataset directory.

    Parameters
    ----------
    dataset_dir
        Directory containing the dataset.
    options
        Optional DatasetScanOptions to configure scanning.

    Returns
    -------
    pyarrow.RecordBatchReader | None
        Reader for the dataset, or ``None`` if unavailable.
    """
    if not dataset_dir.is_dir():
        return None
    try:
        dataset = ds.dataset(str(dataset_dir), format="parquet")
        resolved = options or DatasetScanOptions(batch_size=DEFAULT_ARROW_BATCH_SIZE)
        scanner = build_scanner(dataset, options=resolved)
        return scanner.to_reader()
    except (OSError, ValueError, pa.ArrowInvalid):
        return None


def scan_dataset_lazyframe(
    dataset_dir: Path,
    *,
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
) -> PolarsLazyFrame | None:
    """Return a Polars LazyFrame for a dataset directory.

    Parameters
    ----------
    dataset_dir
        Directory containing the dataset.
    batch_size
        Target batch size for the lazy scan.
    row_index_name
        Optional row index column to attach.
    row_index_offset
        Optional offset for row index values.

    Returns
    -------
    polars.LazyFrame | None
        LazyFrame scan of the dataset, or ``None`` if unavailable.
    """
    if pl is None:  # pragma: no cover - optional dependency
        return None
    if not dataset_dir.is_dir():
        return None
    configure_arrow_threading()
    try:
        if row_index_name:
            return _scan_parquet_with_row_index(
                dataset_dir,
                row_index_name=row_index_name,
                row_index_offset=row_index_offset,
            )
        dataset = ds.dataset(str(dataset_dir), format="parquet")
        return pl.scan_pyarrow_dataset(dataset, batch_size=batch_size)
    except (OSError, ValueError, pa.ArrowInvalid):
        return None


def sample_reader(
    reader: pa.RecordBatchReader,
    *,
    max_rows: int,
) -> pa.RecordBatchReader:
    """Return a reader truncated to a maximum number of rows.

    Parameters
    ----------
    reader
        Source record batch reader.
    max_rows
        Maximum number of rows to include.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader limited to the requested row count.
    """
    if max_rows <= 0:
        return empty_reader_from_schema(reader.schema)

    def _iter_batches() -> Iterable[pa.RecordBatch]:
        remaining = max_rows
        for batch in reader:
            if remaining <= 0:
                break
            current = batch
            if current.num_rows > remaining:
                current = current.slice(0, remaining)
            remaining -= current.num_rows
            yield current

    return pa.RecordBatchReader.from_batches(reader.schema, _iter_batches())


def _dataset_fragments(dataset: ds.Dataset) -> Iterable[ds.Fragment] | None:
    get_fragments = getattr(dataset, "get_fragments", None)
    if not callable(get_fragments):
        return None
    try:
        fragments = cast("Callable[[], Iterable[ds.Fragment]]", get_fragments)()
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None
    try:
        return tuple(fragments)
    except TypeError:
        return None


def _scanner_with_schema(dataset: ds.Dataset, scan_kwargs: dict[str, object]) -> Scanner:
    try:
        return dataset.scanner(**scan_kwargs)
    except TypeError:
        scan_kwargs.pop("schema", None)
        return dataset.scanner(**scan_kwargs)


def _resolve_scan_schema(
    dataset: ds.Dataset,
    options: DatasetScanOptions,
) -> pa.Schema | None:
    schema = options.schema
    if schema is None and options.unify_schemas:
        return unify_dataset_schema(
            dataset,
            schema_promote_options=options.schema_promote_options,
        )
    return schema


def _build_scan_kwargs(
    options: DatasetScanOptions,
    schema: pa.Schema | None,
) -> dict[str, object]:
    scan_kwargs: dict[str, object] = {"batch_size": options.batch_size}
    if options.batch_readahead is not None:
        scan_kwargs["batch_readahead"] = options.batch_readahead
    if options.fragment_readahead is not None:
        scan_kwargs["fragment_readahead"] = options.fragment_readahead
    if options.use_threads is not None:
        scan_kwargs["use_threads"] = options.use_threads
    if options.memory_pool is not None:
        scan_kwargs["memory_pool"] = options.memory_pool
    if options.columns is not None:
        scan_kwargs["columns"] = list(options.columns)
    if schema is not None:
        scan_kwargs["schema"] = schema
    return scan_kwargs


def _scanner_from_fragments(
    fragments: tuple[ds.Fragment, ...] | None,
    schema: pa.Schema,
    scan_kwargs: dict[str, object],
    *,
    filter_expression: ds.Expression | None,
) -> Scanner | None:
    if not fragments:
        return None
    from_fragments = getattr(ds.Scanner, "from_fragments", None)
    scanner_kwargs = dict(scan_kwargs)
    if filter_expression is not None:
        scanner_kwargs["filter"] = filter_expression
    if callable(from_fragments):
        try:
            return from_fragments(fragments, schema=schema, **scanner_kwargs)
        except (TypeError, ValueError, pa.ArrowInvalid):
            return None
    from_fragment = getattr(ds.Scanner, "from_fragment", None)
    if callable(from_fragment) and len(fragments) == 1:
        try:
            return from_fragment(fragments[0], schema=schema, **scanner_kwargs)
        except (TypeError, ValueError, pa.ArrowInvalid):
            return None
    return None


def _fragments_for_filter(
    dataset: ds.Dataset,
    filter_expression: ds.Expression,
) -> tuple[ds.Fragment, ...] | None:
    get_fragments = getattr(dataset, "get_fragments", None)
    if not callable(get_fragments):
        return None
    get_fragments_fn = cast("Callable[..., Iterable[ds.Fragment]]", get_fragments)
    fragments = _safe_get_fragments(get_fragments_fn, filter_expression)
    if fragments is None:
        fragments = _safe_get_fragments(get_fragments_fn, None)
    if fragments is None:
        return None
    return apply_row_group_pruning(fragments, filter_expression)


def _dataset_metadata_path(dataset_dir: Path) -> Path | None:
    metadata_path = dataset_dir / _METADATA_FILENAME
    return metadata_path if metadata_path.is_file() else None


def _common_metadata_path(dataset_dir: Path) -> Path | None:
    metadata_path = dataset_dir / _COMMON_METADATA_FILENAME
    return metadata_path if metadata_path.is_file() else None


def _schema_from_common_metadata(dataset_dir: Path) -> pa.Schema | None:
    common_metadata_path = _common_metadata_path(dataset_dir)
    if common_metadata_path is None:
        return None
    try:
        parquet_file = pq.ParquetFile(common_metadata_path)
    except (OSError, ValueError, pa.ArrowInvalid):
        return None
    return parquet_file.schema_arrow


def _parquet_format_for_manifest(
    extras: Mapping[str, object] | None,
    *,
    schema: pa.Schema | None,
) -> ds.ParquetFileFormat:
    parquet_format = ds.ParquetFileFormat()
    dictionary_columns = _dictionary_columns_from_manifest(extras)
    if dictionary_columns:
        if schema is not None:
            dictionary_columns = tuple(
                name for name in dictionary_columns if name in schema.names
            )
            if not dictionary_columns:
                return parquet_format
        read_options = getattr(parquet_format, "read_options", None)
        if read_options is not None:
            with contextlib.suppress(AttributeError, TypeError, ValueError):
                read_options.dictionary_columns = set(dictionary_columns)
    return parquet_format


def _dataset_from_metadata(
    metadata_path: Path,
    *,
    dataset_dir: Path,
    partitioning: ds.Partitioning | str | None,
    schema: pa.Schema | None,
    parquet_format: ds.FileFormat | None,
) -> ds.Dataset | None:
    try:
        return ds.parquet_dataset(
            str(metadata_path),
            format=parquet_format,
            partitioning=partitioning,
            partition_base_dir=str(dataset_dir),
            schema=schema,
        )
    except (OSError, ValueError, pa.ArrowInvalid):
        return None


def _dictionary_columns_from_manifest(
    extras: Mapping[str, object] | None,
) -> tuple[str, ...] | None:
    if not extras:
        return None
    write_settings = _coerce_mapping(extras.get("write_settings"))
    inferred_settings = _coerce_mapping(extras.get("inferred_settings"))
    columns = _read_str_list(write_settings.get("dictionary_encode_columns"))
    if not columns:
        columns = _read_str_list(inferred_settings.get("dictionary_encode_columns"))
    if columns:
        return tuple(columns)
    return None


def _coerce_mapping(value: object | None) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): val for key, val in value.items()}
    return {}


def _read_str_list(value: object) -> list[str] | None:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return None


def _safe_get_fragments(
    get_fragments: Callable[..., Iterable[ds.Fragment]],
    filter_expression: ds.Expression | None,
) -> tuple[ds.Fragment, ...] | None:
    if filter_expression is None:
        try:
            fragments = get_fragments()
        except (TypeError, ValueError, pa.ArrowInvalid):
            return None
    else:
        try:
            fragments = get_fragments(filter=filter_expression)
        except (TypeError, ValueError, pa.ArrowInvalid):
            return None
    try:
        return tuple(fragments)
    except TypeError:
        return None


def apply_row_group_pruning(
    fragments: tuple[ds.Fragment, ...],
    filter_expression: ds.Expression,
) -> tuple[ds.Fragment, ...]:
    """Apply row-group pruning for fragments using the filter expression.

    Returns
    -------
    tuple[pyarrow.dataset.Fragment, ...]
        Fragments filtered by row-group metadata when supported.
    """
    pruned: list[ds.Fragment] = []
    for fragment in fragments:
        subset = getattr(fragment, "subset", None)
        if not callable(subset):
            pruned.append(fragment)
            continue
        try:
            subset_value = subset(filter_expression)
        except (TypeError, ValueError, pa.ArrowInvalid):
            pruned.append(fragment)
            continue
        if subset_value is None:
            continue
        if isinstance(subset_value, ds.Fragment):
            pruned.append(subset_value)
            continue
        try:
            iterator = iter(cast("Iterable[ds.Fragment]", subset_value))
        except TypeError:
            pruned.append(fragment)
            continue
        filtered = [item for item in iterator if isinstance(item, ds.Fragment)]
        if filtered:
            pruned.extend(filtered)
        else:
            pruned.append(fragment)
    return tuple(pruned)


def _scan_parquet_with_row_index(
    dataset_dir: Path,
    *,
    row_index_name: str,
    row_index_offset: int,
) -> PolarsLazyFrame:
    if pl is None:  # pragma: no cover - defensive guard
        msg = "polars is required for parquet scans"
        raise RuntimeError(msg)
    scan_fn = cast("Callable[..., PolarsLazyFrame]", pl.scan_parquet)
    try:
        signature = inspect.signature(scan_fn)
    except (TypeError, ValueError):
        return scan_fn(str(dataset_dir), row_index_name=row_index_name)
    if "row_index_offset" in signature.parameters:
        return scan_fn(
            str(dataset_dir),
            row_index_name=row_index_name,
            row_index_offset=row_index_offset,
        )
    return scan_fn(str(dataset_dir), row_index_name=row_index_name)


__all__ = [
    "DatasetScanOptions",
    "QueryPlanSpec",
    "apply_row_group_pruning",
    "build_scanner",
    "configure_arrow_threading",
    "dataset_for_manifest",
    "empty_reader_from_schema",
    "resolve_partitioning",
    "sample_reader",
    "scan_dataset_lazyframe",
    "scan_dataset_reader",
    "unify_dataset_schema",
]
