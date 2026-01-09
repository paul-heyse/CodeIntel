"""Arrow-first streaming helpers for datasets and readers."""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.columnar import profiles as columnar_profiles
from codeintel.core.columnar.conversion import record_batch_reader_from_iterable
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_arrow_scan_settings,
    resolve_execution_context,
)
from codeintel.core.columnar.profiles import DatasetScanOptions, scan_profile_options
from codeintel.core.columnar.queryspec import QuerySpec
from codeintel.core.columnar.readers import empty_reader_from_schema
from codeintel.core.columnar.runtime import apply_runtime_profile
from codeintel.core.columnar.schema import DEFAULT_SCHEMA_PROMOTE_OPTIONS, SchemaPromoteOptions
from codeintel.core.columnar.schema_ops import unify_schemas
from codeintel.core.config.settings import ArrowScanSettings
from codeintel.core.constants import (
    DEFAULT_ARROW_BATCH_READAHEAD,
    DEFAULT_ARROW_BATCH_SIZE,
    DEFAULT_ARROW_CACHE_METADATA,
    DEFAULT_ARROW_CPU_COUNT,
    DEFAULT_ARROW_FRAGMENT_READAHEAD,
    DEFAULT_ARROW_IO_THREAD_COUNT,
    DEFAULT_ARROW_PARQUET_BUFFER_SIZE,
    DEFAULT_ARROW_PARQUET_PRE_BUFFER,
    DEFAULT_ARROW_PARQUET_USE_BUFFERED_STREAM,
    DEFAULT_ARROW_USE_THREADS,
)
from codeintel.core.datasets.parquet_metadata import DatasetMetadataContext
from codeintel.core.manifests import ArrowDatasetManifest

if TYPE_CHECKING:
    from polars import LazyFrame
    from pyarrow.dataset import Scanner

    from codeintel.core.columnar.profiles import RuntimeProfile

    type PolarsLazyFrame = LazyFrame
else:
    type PolarsLazyFrame = object

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None

ScanProfile = columnar_profiles.ScanProfile


@dataclass(frozen=True, slots=True)
class ScanTelemetry:
    """Scan telemetry captured during dataset planning."""

    fragment_count: int | None
    estimated_rows: int | None


_DATASET_IGNORE_PREFIXES: tuple[str, ...] = (".", "_")
_DATASET_EXCLUDE_INVALID_FILES = True


def _resolve_arrow_scan_settings(
    ctx: ExecutionContext | None,
    settings: ArrowScanSettings | None,
) -> ArrowScanSettings:
    if settings is not None:
        return settings
    return resolve_arrow_scan_settings(ctx)


def _override_default_int(current: int, override: int, *, default: int) -> int:
    return override if current == default else current


def _override_default_optional(
    current: int | None,
    override: int | None,
    *,
    default: int | None,
) -> int | None:
    if override is None:
        return current
    if current is None or current == default:
        return override
    return current


def _override_default_bool(
    *,
    current: bool | None,
    override: bool | None,
    default: bool | None,
) -> bool | None:
    if override is None:
        return current
    if current is None or current == default:
        return override
    return current


def _merge_scan_options(
    options: DatasetScanOptions,
    settings: ArrowScanSettings,
) -> DatasetScanOptions:
    resolved = replace(
        options,
        batch_size=_override_default_int(
            options.batch_size,
            settings.batch_size,
            default=DEFAULT_ARROW_BATCH_SIZE,
        ),
        batch_readahead=_override_default_optional(
            options.batch_readahead,
            settings.batch_readahead,
            default=DEFAULT_ARROW_BATCH_READAHEAD,
        ),
        fragment_readahead=_override_default_optional(
            options.fragment_readahead,
            settings.fragment_readahead,
            default=DEFAULT_ARROW_FRAGMENT_READAHEAD,
        ),
        cache_metadata=_override_default_bool(
            current=options.cache_metadata,
            override=settings.cache_metadata,
            default=DEFAULT_ARROW_CACHE_METADATA,
        ),
        use_threads=_override_default_bool(
            current=options.use_threads,
            override=settings.use_threads,
            default=DEFAULT_ARROW_USE_THREADS,
        ),
        parquet_pre_buffer=_override_default_bool(
            current=options.parquet_pre_buffer,
            override=settings.parquet_pre_buffer,
            default=DEFAULT_ARROW_PARQUET_PRE_BUFFER,
        ),
        parquet_use_buffered_stream=_override_default_bool(
            current=options.parquet_use_buffered_stream,
            override=settings.parquet_use_buffered_stream,
            default=DEFAULT_ARROW_PARQUET_USE_BUFFERED_STREAM,
        ),
        parquet_buffer_size=_override_default_optional(
            options.parquet_buffer_size,
            settings.parquet_buffer_size,
            default=DEFAULT_ARROW_PARQUET_BUFFER_SIZE,
        ),
    )
    return resolved


def configure_arrow_threading(
    *,
    cpu_count: int | None = DEFAULT_ARROW_CPU_COUNT,
    io_thread_count: int | None = DEFAULT_ARROW_IO_THREAD_COUNT,
    settings: ArrowScanSettings | None = None,
) -> None:
    """Apply Arrow threading defaults for compute and dataset scans."""
    resolved_ctx = resolve_execution_context(None)
    resolved_settings = _resolve_arrow_scan_settings(resolved_ctx, settings)
    profile = resolved_ctx.runtime_profile
    override_cpu = None if cpu_count == DEFAULT_ARROW_CPU_COUNT else cpu_count
    override_io = None if io_thread_count == DEFAULT_ARROW_IO_THREAD_COUNT else io_thread_count
    if override_cpu is not None or override_io is not None:
        if profile is None:
            profile = columnar_profiles.RuntimeProfile(
                cpu_threads=override_cpu,
                io_threads=override_io,
            )
        else:
            profile = replace(
                profile,
                cpu_threads=override_cpu if override_cpu is not None else profile.cpu_threads,
                io_threads=override_io if override_io is not None else profile.io_threads,
            )
    apply_runtime_profile(profile, settings=resolved_settings)


def configure_arrow_threading_for_context(
    *,
    ctx: ExecutionContext | None,
    settings: ArrowScanSettings | None = None,
) -> None:
    """Apply Arrow threading defaults using an execution context profile.

    Parameters
    ----------
    ctx
        Execution context providing runtime profile defaults.
    settings
        Optional Arrow scan settings override.
    """
    resolved_ctx = resolve_execution_context(ctx)
    profile = resolved_ctx.runtime_profile
    resolved_settings = _resolve_arrow_scan_settings(resolved_ctx, settings)
    apply_runtime_profile(profile, settings=resolved_settings)


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
    columns = [str(column) for column in manifest.partition_columns]
    if schema is None:
        return ds.partitioning(field_names=columns)
    if any(column not in schema.names for column in columns):
        return ds.partitioning(field_names=columns)
    fields: list[pa.Field] = []
    for name in columns:
        field = schema.field(name)
        if pa.types.is_dictionary(field.type):
            field = pa.field(name, field.type.value_type, field.nullable)
        fields.append(field)
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
    metadata_ctx = DatasetMetadataContext(
        dataset_root=dataset_dir,
        table_key=manifest.table_key,
    )
    metadata_path = metadata_ctx.metadata_path()
    common_metadata_path = metadata_ctx.common_metadata_path()
    schema = metadata_ctx.read_schema()
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
        dataset_kwargs = _dataset_kwargs(
            file_format=parquet_format,
            partitioning=partitioning,
            schema=schema,
        )
        return ds.dataset(paths, **dataset_kwargs)
    dataset_kwargs = _dataset_kwargs(
        file_format=parquet_format,
        partitioning=partitioning,
        schema=schema,
    )
    return ds.dataset(str(dataset_dir), **dataset_kwargs)


def dataset_for_path(
    dataset_dir: Path,
    *,
    schema: pa.Schema | None = None,
) -> ds.Dataset:
    """Return a PyArrow dataset for a parquet directory path.

    Parameters
    ----------
    dataset_dir
        Directory containing dataset files.
    schema
        Optional schema override for dataset discovery.

    Returns
    -------
    pyarrow.dataset.Dataset
        Dataset constructed for the directory path.
    """
    dataset_kwargs = _dataset_kwargs(file_format="parquet", schema=schema)
    return ds.dataset(str(dataset_dir), **dataset_kwargs)


def build_scanner(
    dataset: ds.Dataset,
    *,
    options: DatasetScanOptions,
    ctx: ExecutionContext | None = None,
) -> Scanner:
    """Build a dataset scanner using shared scan options.

    Parameters
    ----------
    dataset
        Arrow dataset to scan.
    options
        Scan configuration options.
    ctx
        Optional execution context for runtime profile defaults.

    Returns
    -------
    pyarrow.dataset.Scanner
        Scanner configured for the dataset.
    """
    resolved_ctx = resolve_execution_context(ctx)
    scan_settings = _resolve_arrow_scan_settings(resolved_ctx, None)
    resolved_options = _merge_scan_options(options, scan_settings)
    resolved_options = _apply_runtime_profile_scan_defaults(
        resolved_options,
        profile=resolved_ctx.runtime_profile,
    )
    configure_arrow_threading_for_context(ctx=resolved_ctx)
    schema = _resolve_scan_schema(dataset, resolved_options)
    fragment_scan_options = _parquet_fragment_scan_options(dataset, resolved_options)
    scan_kwargs = _build_scan_kwargs(
        resolved_options,
        schema,
        fragment_scan_options=fragment_scan_options,
    )
    filter_expression = resolved_options.filter_expression
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


def scan_telemetry(
    dataset: ds.Dataset,
    *,
    filter_expression: ds.Expression | None,
) -> ScanTelemetry:
    """Return scan telemetry for the provided dataset and filter.

    Returns
    -------
    ScanTelemetry
        Fragment count and estimated rows for the scan.
    """
    fragments = _fragments_for_filter(dataset, filter_expression)
    fragment_count = len(fragments) if fragments is not None else None
    try:
        estimated_rows = dataset.count_rows(filter=filter_expression)
    except (OSError, ValueError, pa.ArrowInvalid):
        estimated_rows = None
    return ScanTelemetry(fragment_count=fragment_count, estimated_rows=estimated_rows)


def scan_options_for_queryspec(
    spec: QuerySpec,
    *,
    provenance: bool,
    options: DatasetScanOptions | None = None,
) -> DatasetScanOptions:
    """Return DatasetScanOptions configured from a QuerySpec.

    Returns
    -------
    DatasetScanOptions
        Scan options reflecting the query specification.
    """
    resolved = options or DatasetScanOptions()
    columns = spec.scan_columns(provenance=provenance)
    filter_expression = spec.scan_filter_expression()
    return replace(
        resolved,
        columns=columns,
        filter_expression=filter_expression,
    )


def scan_options_for_queryspec_ctx(
    spec: QuerySpec,
    *,
    ctx: ExecutionContext | None,
    options: DatasetScanOptions | None = None,
) -> DatasetScanOptions:
    """Return DatasetScanOptions configured from a QuerySpec and execution context.

    Returns
    -------
    DatasetScanOptions
        Scan options reflecting query spec and execution context.
    """
    resolved_ctx = resolve_execution_context(ctx)
    resolved = options or DatasetScanOptions()
    profile = resolved_ctx.runtime_profile
    resolved = _apply_runtime_profile_scan_defaults(resolved, profile=profile)
    provenance = resolved_ctx.provenance
    if profile is not None:
        provenance = profile.resolve_provenance(default=provenance)
    determinism = resolved_ctx.resolve_determinism()
    if determinism == "canonical":
        provenance = True
    return scan_options_for_queryspec(
        spec,
        provenance=provenance,
        options=resolved,
    )


def _apply_runtime_profile_scan_defaults(
    options: DatasetScanOptions,
    *,
    profile: RuntimeProfile | None,
) -> DatasetScanOptions:
    if profile is None:
        return options
    resolved = options
    scan_profile = profile.resolve_scan_profile(default=None)
    if scan_profile:
        resolved = scan_profile_options(scan_profile, base=resolved)
    implicit_ordering = profile.resolve_implicit_ordering(default=resolved.implicit_ordering)
    require_sequenced_output = profile.resolve_require_sequenced_output(
        default=resolved.require_sequenced_output
    )
    use_threads = profile.resolve_scan_use_threads(default=resolved.use_threads)
    if (
        implicit_ordering == resolved.implicit_ordering
        and require_sequenced_output == resolved.require_sequenced_output
        and use_threads == resolved.use_threads
    ):
        return resolved
    return replace(
        resolved,
        implicit_ordering=implicit_ordering,
        require_sequenced_output=require_sequenced_output,
        use_threads=use_threads,
    )


def build_scanner_for_queryspec(
    dataset: ds.Dataset,
    *,
    spec: QuerySpec,
    provenance: bool,
    options: DatasetScanOptions | None = None,
) -> Scanner:
    """Build a dataset scanner from a QuerySpec.

    Returns
    -------
    pyarrow.dataset.Scanner
        Scanner configured from the query specification.
    """
    scan_options = scan_options_for_queryspec(
        spec,
        provenance=provenance,
        options=options,
    )
    return build_scanner(dataset, options=scan_options)


def build_scanner_for_queryspec_ctx(
    dataset: ds.Dataset,
    *,
    spec: QuerySpec,
    ctx: ExecutionContext | None,
    options: DatasetScanOptions | None = None,
) -> Scanner:
    """Build a dataset scanner from a QuerySpec and execution context.

    Returns
    -------
    pyarrow.dataset.Scanner
        Scanner configured from query spec and execution context.
    """
    configure_arrow_threading_for_context(ctx=ctx)
    scan_options = scan_options_for_queryspec_ctx(
        spec,
        ctx=ctx,
        options=options,
    )
    return build_scanner(dataset, options=scan_options, ctx=ctx)


def scan_telemetry_for_queryspec(
    dataset: ds.Dataset,
    *,
    spec: QuerySpec,
) -> ScanTelemetry:
    """Return scan telemetry based on a QuerySpec.

    Returns
    -------
    ScanTelemetry
        Fragment count and estimated rows for the query.
    """
    filter_expression = spec.scan_filter_expression()
    return scan_telemetry(dataset, filter_expression=filter_expression)


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


def scan_dataset_reader(
    dataset_dir: Path,
    *,
    options: DatasetScanOptions | None = None,
    ctx: ExecutionContext | None = None,
) -> pa.RecordBatchReader | None:
    """Return a streaming reader for a dataset directory.

    Parameters
    ----------
    dataset_dir
        Directory containing the dataset.
    options
        Optional DatasetScanOptions to configure scanning.
    ctx
        Optional execution context for runtime profile defaults.

    Returns
    -------
    pyarrow.RecordBatchReader | None
        Reader for the dataset, or ``None`` if unavailable.
    """
    if not dataset_dir.is_dir():
        return None
    try:
        dataset = ds.dataset(str(dataset_dir), **_dataset_kwargs(file_format="parquet"))
        resolved = options or DatasetScanOptions()
        scanner = build_scanner(dataset, options=resolved, ctx=ctx)
        return scanner.to_reader()
    except (OSError, ValueError, pa.ArrowInvalid):
        return None


def scan_dataset_lazyframe(
    dataset_dir: Path,
    *,
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    ctx: ExecutionContext | None = None,
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
    ctx
        Optional execution context for runtime profile defaults.

    Returns
    -------
    polars.LazyFrame | None
        LazyFrame scan of the dataset, or ``None`` if unavailable.
    """
    if pl is None:  # pragma: no cover - optional dependency
        return None
    if not dataset_dir.is_dir():
        return None
    resolved_ctx = resolve_execution_context(ctx)
    scan_settings = _resolve_arrow_scan_settings(resolved_ctx, None)
    resolved_options = _merge_scan_options(
        DatasetScanOptions(batch_size=batch_size),
        scan_settings,
    )
    resolved_options = _apply_runtime_profile_scan_defaults(
        resolved_options,
        profile=resolved_ctx.runtime_profile,
    )
    resolved_batch_size = resolved_options.batch_size
    configure_arrow_threading_for_context(ctx=resolved_ctx)
    try:
        if row_index_name:
            return _scan_parquet_with_row_index(
                dataset_dir,
                row_index_name=row_index_name,
                row_index_offset=row_index_offset,
            )
        dataset = ds.dataset(str(dataset_dir), **_dataset_kwargs(file_format="parquet"))
        return pl.scan_pyarrow_dataset(dataset, batch_size=resolved_batch_size)
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

    sampled = record_batch_reader_from_iterable(_iter_batches(), empty_policy="none")
    if sampled is None:
        return empty_reader_from_schema(reader.schema)
    return sampled


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
    scan_kwargs = _filter_scan_kwargs(dataset.scanner, scan_kwargs)
    try:
        return dataset.scanner(**scan_kwargs)
    except TypeError:
        scan_kwargs.pop("schema", None)
        scan_kwargs = _filter_scan_kwargs(dataset.scanner, scan_kwargs)
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


def _filter_scan_kwargs(
    target: Callable[..., object], kwargs: dict[str, object]
) -> dict[str, object]:
    return _filter_kwargs(target, kwargs)


def _filter_kwargs(target: Callable[..., object], kwargs: dict[str, object]) -> dict[str, object]:
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in params}


def _dataset_is_parquet(dataset: ds.Dataset) -> bool:
    dataset_format = getattr(dataset, "format", None)
    return isinstance(dataset_format, ds.ParquetFileFormat)


def _parquet_fragment_scan_options(
    dataset: ds.Dataset,
    options: DatasetScanOptions,
) -> object | None:
    parquet_options = getattr(ds, "ParquetFragmentScanOptions", None)
    if not callable(parquet_options):
        return None
    if not _dataset_is_parquet(dataset):
        return None
    kwargs: dict[str, object] = {}
    if options.parquet_pre_buffer is not None:
        kwargs["pre_buffer"] = options.parquet_pre_buffer
    if options.parquet_use_buffered_stream is not None:
        kwargs["use_buffered_stream"] = options.parquet_use_buffered_stream
    if options.parquet_buffer_size is not None:
        kwargs["buffer_size"] = options.parquet_buffer_size
    if not kwargs:
        return None
    filtered = _filter_kwargs(parquet_options, kwargs)
    if not filtered:
        return None
    try:
        return parquet_options(**filtered)
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None


def _build_scan_kwargs(
    options: DatasetScanOptions,
    schema: pa.Schema | None,
    *,
    fragment_scan_options: object | None,
) -> dict[str, object]:
    scan_kwargs: dict[str, object] = {"batch_size": options.batch_size}
    optional_kwargs: dict[str, object | None] = {
        "batch_readahead": options.batch_readahead,
        "fragment_readahead": options.fragment_readahead,
        "cache_metadata": options.cache_metadata,
        "use_threads": options.use_threads,
        "memory_pool": options.memory_pool,
        "implicit_ordering": options.implicit_ordering,
        "require_sequenced_output": options.require_sequenced_output,
        "schema": schema,
        "fragment_scan_options": fragment_scan_options,
    }
    scan_kwargs.update({key: value for key, value in optional_kwargs.items() if value is not None})
    columns = options.projection_columns()
    if columns is not None:
        if isinstance(columns, Mapping):
            scan_kwargs["columns"] = columns
        else:
            scan_kwargs["columns"] = list(columns)
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
            scanner_kwargs = _filter_scan_kwargs(from_fragments, scanner_kwargs)
            return from_fragments(fragments, schema=schema, **scanner_kwargs)
        except (TypeError, ValueError, pa.ArrowInvalid):
            return None
    from_fragment = getattr(ds.Scanner, "from_fragment", None)
    if callable(from_fragment) and len(fragments) == 1:
        try:
            scanner_kwargs = _filter_scan_kwargs(from_fragment, scanner_kwargs)
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


def _parquet_format_for_manifest(
    extras: Mapping[str, object] | None,
    *,
    schema: pa.Schema | None,
) -> ds.ParquetFileFormat:
    parquet_format = ds.ParquetFileFormat()
    dictionary_columns = _dictionary_columns_from_manifest(extras)
    if dictionary_columns:
        if schema is not None:
            dictionary_columns = tuple(name for name in dictionary_columns if name in schema.names)
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
    dataset_kwargs = _parquet_dataset_kwargs(
        partitioning=partitioning,
        schema=schema,
        parquet_format=parquet_format,
        dataset_dir=dataset_dir,
    )
    try:
        return ds.parquet_dataset(
            str(metadata_path),
            **dataset_kwargs,
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


def _dataset_discovery_kwargs() -> dict[str, object]:
    return {
        "exclude_invalid_files": _DATASET_EXCLUDE_INVALID_FILES,
        "ignore_prefixes": list(_DATASET_IGNORE_PREFIXES),
    }


def _dataset_kwargs(
    *,
    file_format: ds.FileFormat | None = None,
    partitioning: ds.Partitioning | str | None = None,
    schema: pa.Schema | None = None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "format": file_format,
        "partitioning": partitioning,
        "schema": schema,
    }
    kwargs.update(_dataset_discovery_kwargs())
    return _filter_kwargs(ds.dataset, kwargs)


def _parquet_dataset_kwargs(
    *,
    dataset_dir: Path,
    parquet_format: ds.FileFormat | None,
    partitioning: ds.Partitioning | str | None,
    schema: pa.Schema | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "format": parquet_format,
        "partitioning": partitioning,
        "partition_base_dir": str(dataset_dir),
        "schema": schema,
    }
    kwargs.update(_dataset_discovery_kwargs())
    return _filter_kwargs(ds.parquet_dataset, kwargs)


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
    "ScanProfile",
    "ScanTelemetry",
    "apply_row_group_pruning",
    "build_scanner",
    "build_scanner_for_queryspec",
    "build_scanner_for_queryspec_ctx",
    "configure_arrow_threading",
    "configure_arrow_threading_for_context",
    "dataset_for_manifest",
    "dataset_for_path",
    "empty_reader_from_schema",
    "resolve_partitioning",
    "sample_reader",
    "scan_dataset_lazyframe",
    "scan_dataset_reader",
    "scan_options_for_queryspec",
    "scan_options_for_queryspec_ctx",
    "scan_profile_options",
    "scan_telemetry",
    "scan_telemetry_for_queryspec",
    "unify_dataset_schema",
]
