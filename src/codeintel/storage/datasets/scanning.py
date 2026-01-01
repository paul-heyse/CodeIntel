"""Shared Arrow dataset scanning helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from codeintel.core.manifests import ArrowDatasetManifest

if TYPE_CHECKING:
    from pyarrow.dataset import Scanner


@dataclass(frozen=True, slots=True)
class DatasetScanOptions:
    """Options for Arrow dataset scanning."""

    batch_size: int
    batch_readahead: int | None = None
    fragment_readahead: int | None = None
    filter_expression: ds.Expression | None = None
    use_threads: bool | None = None
    memory_pool: pa.MemoryPool | None = None
    schema: pa.Schema | None = None
    columns: Sequence[str] | None = None
    unify_schemas: bool = False
    metrics_enabled: bool = False


@dataclass(frozen=True, slots=True)
class QueryPlanSpec:
    """Shared query plan details for dataset scanning."""

    table_key: str
    columns: tuple[str, ...]
    filter_expression: ds.Expression | None


_METADATA_FILENAME = "_metadata"
_COMMON_METADATA_FILENAME = "_common_metadata"


def resolve_partitioning(
    *,
    manifest: ArrowDatasetManifest,
    schema: pa.Schema | None,
) -> ds.Partitioning | str | None:
    """Resolve dataset partitioning from manifest metadata.

    Returns
    -------
    pyarrow.dataset.Partitioning | str | None
        Partitioning config for dataset reads.
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

    Returns
    -------
    pyarrow.dataset.Dataset
        Dataset handle for the manifest.
    """
    dataset_dir = manifest_path.parent.resolve()
    metadata_path = _dataset_metadata_path(dataset_dir)
    common_metadata_path = _common_metadata_path(dataset_dir)
    schema = _schema_from_common_metadata(dataset_dir)
    partitioning = resolve_partitioning(manifest=manifest, schema=schema)
    metadata_source = metadata_path or common_metadata_path
    if metadata_source is not None:
        dataset = _dataset_from_metadata(
            metadata_source,
            dataset_dir=dataset_dir,
            partitioning=partitioning,
            schema=schema,
        )
        if dataset is not None:
            return dataset
    if manifest.files:
        paths = [str(dataset_dir / path) for path in manifest.files]
        return ds.dataset(paths, format="parquet", partitioning=partitioning, schema=schema)
    return ds.dataset(str(dataset_dir), format="parquet", partitioning=partitioning, schema=schema)


def build_scanner(dataset: ds.Dataset, *, options: DatasetScanOptions) -> Scanner:
    """Build a dataset scanner using shared scan options.

    Returns
    -------
    Scanner
        Configured scanner for the dataset.
    """
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


def unify_dataset_schema(dataset: ds.Dataset) -> pa.Schema | None:
    """Return a unified schema for a dataset when fragments diverge.

    Returns
    -------
    pa.Schema | None
        Unified schema if available; otherwise dataset schema.
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
        return pa.unify_schemas(schemas)
    except (TypeError, ValueError, pa.ArrowInvalid):
        return dataset.schema


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
        return unify_dataset_schema(dataset)
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
    return _apply_row_group_pruning(fragments, filter_expression)


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


def _dataset_from_metadata(
    metadata_path: Path,
    *,
    dataset_dir: Path,
    partitioning: ds.Partitioning | str | None,
    schema: pa.Schema | None,
) -> ds.Dataset | None:
    try:
        return ds.parquet_dataset(
            str(metadata_path),
            partitioning=partitioning,
            partition_base_dir=str(dataset_dir),
            schema=schema,
        )
    except (OSError, ValueError, pa.ArrowInvalid):
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


def _apply_row_group_pruning(
    fragments: tuple[ds.Fragment, ...],
    filter_expression: ds.Expression,
) -> tuple[ds.Fragment, ...]:
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


__all__ = [
    "DatasetScanOptions",
    "build_scanner",
    "dataset_for_manifest",
    "resolve_partitioning",
    "unify_dataset_schema",
]
