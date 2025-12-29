"""Arrow dataset manifest helpers for serving engines."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.serving.semantic.filter_ops import allowed_ops_for_column_type
from codeintel.storage.datasets.manifests import read_dataset_manifest

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.manifests import ArrowDatasetManifest, ServingSnapshotManifest
    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.semantic.models import FilterSpec, FilterValue

LOG = logging.getLogger(__name__)


_COMPARISON_OPS = frozenset({"eq", "ne", "lt", "lte", "gt", "gte"})
_STRING_OPS = frozenset({"contains", "startswith"})


@dataclass(frozen=True, slots=True)
class DatasetManifestEntry:
    """Dataset manifest plus its on-disk location."""

    manifest: ArrowDatasetManifest
    manifest_path: Path

    @property
    def dataset_dir(self) -> Path:
        """Return the dataset directory for this manifest.

        Returns
        -------
        pathlib.Path
            Directory containing the dataset files.
        """
        return self.manifest_path.parent


@dataclass(frozen=True, slots=True)
class DatasetManifestIndex:
    """Index of dataset manifests keyed by table_key."""

    by_table_key: Mapping[str, DatasetManifestEntry]

    def get(self, table_key: str) -> DatasetManifestEntry | None:
        """Return the dataset manifest entry for a table key, if present.

        Returns
        -------
        DatasetManifestEntry | None
            Manifest entry for the table key, if registered.
        """
        return self.by_table_key.get(table_key)

    def table_keys(self) -> tuple[str, ...]:
        """Return all table keys with dataset manifests.

        Returns
        -------
        tuple[str, ...]
            Table keys backed by dataset manifests.
        """
        return tuple(self.by_table_key.keys())


def dataset_for_entry(entry: DatasetManifestEntry) -> ds.Dataset:
    """Return a PyArrow dataset for a manifest entry.

    Returns
    -------
    pyarrow.dataset.Dataset
        Dataset handle for the manifest entry.
    """
    partitioning: str | None = "hive" if entry.manifest.partition_columns else None
    if entry.manifest.files:
        paths = [str(entry.dataset_dir / path) for path in entry.manifest.files]
        return ds.dataset(paths, format="parquet", partitioning=partitioning)
    return ds.dataset(str(entry.dataset_dir), format="parquet", partitioning=partitioning)


def dataset_filter_expression(
    *,
    filters: list[FilterSpec],
    column_types: Mapping[str, ColumnType] | None = None,
) -> ds.Expression | None:
    """Build a dataset filter expression for pushdown.

    Parameters
    ----------
    filters
        Filter specs to translate into dataset expressions.
    column_types
        Optional column type mapping for operator validation.

    Returns
    -------
    pyarrow.dataset.Expression | None
        Filter expression or None when unsupported.
    """
    if not filters:
        return None
    expressions: list[ds.Expression] = []
    for filt in filters:
        expr = _filter_expression(filt, column_types=column_types)
        if expr is not None:
            expressions.append(expr)
    return _combine_expressions(expressions)


def dataset_scanner_for_entry(
    entry: DatasetManifestEntry,
    *,
    batch_size: int,
    fragment_readahead: int | None = None,
    filter_expression: ds.Expression | None = None,
    metrics_enabled: bool = False,
    schema: pa.Schema | None = None,
) -> ds.Scanner:
    """Return a dataset scanner configured for streaming reads.

    Returns
    -------
    pyarrow.dataset.Scanner
        Scanner configured for batched streaming reads.
    """
    start = perf_counter()
    dataset = dataset_for_entry(entry)
    if metrics_enabled:
        _log_scan_metrics(
            entry,
            dataset=dataset,
            filter_expression=filter_expression,
            duration_ms=(perf_counter() - start) * 1000,
            memory_bytes=_total_allocated_bytes(),
        )
    scan_kwargs: dict[str, object] = {"batch_size": batch_size}
    if fragment_readahead is not None:
        scan_kwargs["fragment_readahead"] = fragment_readahead
    return _build_scanner(
        dataset,
        filter_expression=filter_expression,
        scan_kwargs=scan_kwargs,
        schema=schema,
    )


def load_dataset_manifests(
    snapshot_manifest: ServingSnapshotManifest,
) -> DatasetManifestIndex:
    """Load dataset manifests from a snapshot manifest.

    Returns
    -------
    DatasetManifestIndex
        Loaded dataset manifest index keyed by table key.

    Raises
    ------
    ValueError
        If manifest metadata is inconsistent with the snapshot manifest.
    """
    by_table: dict[str, DatasetManifestEntry] = {}
    for table_key, entry in snapshot_manifest.datasets.items():
        manifest_path = Path(entry.manifest_path)
        manifest = read_dataset_manifest(manifest_path)
        if manifest.table_key != table_key:
            msg = f"Dataset manifest table_key mismatch: {table_key} != {manifest.table_key}"
            raise ValueError(msg)
        if entry.schema_hash is None:
            msg = f"Snapshot dataset entry missing schema_hash for {table_key}"
            raise ValueError(msg)
        if manifest.schema_hash is None:
            msg = f"Dataset manifest missing schema_hash for {table_key}"
            raise ValueError(msg)
        if entry.schema_hash != manifest.schema_hash:
            msg = (
                "Dataset manifest schema hash mismatch for "
                f"{table_key}: {entry.schema_hash} != {manifest.schema_hash}"
            )
            raise ValueError(msg)
        by_table[table_key] = DatasetManifestEntry(
            manifest=manifest,
            manifest_path=manifest_path,
        )
    return DatasetManifestIndex(by_table_key=by_table)


def _combine_expressions(expressions: list[ds.Expression]) -> ds.Expression | None:
    if not expressions:
        return None
    combined = expressions[0]
    for expr in expressions[1:]:
        combined &= expr
    return combined


def _filter_expression(
    filt: FilterSpec,
    *,
    column_types: Mapping[str, ColumnType] | None,
) -> ds.Expression | None:
    column_type = column_types.get(filt.column) if column_types is not None else None
    allowed_ops = allowed_ops_for_column_type(column_type)
    if filt.op not in allowed_ops:
        return None
    field = ds.field(filt.column)
    if filt.op in _COMPARISON_OPS:
        return _comparison_expression(field, op=filt.op, value=filt.value)
    if filt.op == "in":
        return _in_expression(field, value=filt.value)
    if filt.op in _STRING_OPS:
        return _string_expression(field, op=filt.op, value=filt.value)
    return None


def _comparison_expression(
    field: ds.Expression,
    *,
    op: str,
    value: FilterValue,
) -> ds.Expression | None:
    if isinstance(value, list):
        return None
    builder = _DATASET_COMPARISON_BUILDERS.get(op)
    if builder is None:
        return None
    return builder(field, value)


def _in_expression(field: ds.Expression, *, value: FilterValue) -> ds.Expression | None:
    values = value if isinstance(value, list) else [value]
    if not values:
        return None
    isin = getattr(field, "isin", None)
    if callable(isin):
        return isin(values)
    return None


def _string_expression(
    field: ds.Expression,
    *,
    op: str,
    value: FilterValue,
) -> ds.Expression | None:
    if isinstance(value, list) or not isinstance(value, str):
        return None
    method = _string_method(field, op=op)
    if method is None:
        return None
    return method(value)


def _log_scan_metrics(
    entry: DatasetManifestEntry,
    *,
    dataset: ds.Dataset,
    filter_expression: ds.Expression | None,
    duration_ms: float,
    memory_bytes: int | None,
) -> None:
    stats = entry.manifest.stats or {}
    row_groups = stats.get("row_groups")
    total_fragments = _count_fragments(dataset, filter_expression=None)
    filtered_fragments = (
        _count_fragments(dataset, filter_expression=filter_expression)
        if filter_expression is not None
        else None
    )
    LOG.info(
        "dataset_scan_metrics table=%s files=%s fragments=%s fragments_filtered=%s "
        "row_groups=%s duration_ms=%.2f memory_bytes=%s",
        entry.manifest.table_key,
        len(entry.manifest.files),
        total_fragments,
        filtered_fragments,
        row_groups,
        duration_ms,
        memory_bytes,
    )


def _count_fragments(
    dataset: ds.Dataset,
    *,
    filter_expression: ds.Expression | None,
) -> int | None:
    get_fragments = getattr(dataset, "get_fragments", None)
    if not callable(get_fragments):
        return None
    try:
        fragments = (
            get_fragments(filter=filter_expression)
            if filter_expression is not None
            else get_fragments()
        )
        if not isinstance(fragments, Iterable):
            return None
        return sum(1 for _ in fragments)
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None


def _total_allocated_bytes() -> int | None:
    total_fn = getattr(pa, "total_allocated_bytes", None)
    if not callable(total_fn):
        return None
    try:
        total = total_fn()
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None
    return _coerce_int(total)


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    result: int | None = None
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        result = int(value)
    elif isinstance(value, str):
        try:
            result = int(value)
        except ValueError:
            result = None
    else:
        converter = getattr(value, "__int__", None)
        if callable(converter):
            try:
                result = int(value)
            except (TypeError, ValueError):
                result = None
    return result


def _build_scanner(
    dataset: ds.Dataset,
    *,
    filter_expression: ds.Expression | None,
    scan_kwargs: Mapping[str, object],
    schema: pa.Schema | None,
) -> ds.Scanner:
    resolved_schema = schema or dataset.schema
    scan_kwargs = dict(scan_kwargs)
    if schema is not None:
        scan_kwargs["schema"] = resolved_schema
    if filter_expression is None:
        return _scanner_with_schema(dataset, scan_kwargs)
    fragments = _fragments_for_filter(dataset, filter_expression)
    from_fragments = getattr(ds.Scanner, "from_fragments", None)
    if fragments is not None and callable(from_fragments):
        try:
            return from_fragments(fragments, schema=resolved_schema, **scan_kwargs)
        except (TypeError, ValueError, pa.ArrowInvalid):
            pass
    scan_kwargs["filter"] = filter_expression
    return _scanner_with_schema(dataset, scan_kwargs)


def _scanner_with_schema(dataset: ds.Dataset, scan_kwargs: dict[str, object]) -> ds.Scanner:
    try:
        return dataset.scanner(**scan_kwargs)
    except TypeError:
        scan_kwargs.pop("schema", None)
        return dataset.scanner(**scan_kwargs)


def _fragments_for_filter(
    dataset: ds.Dataset,
    filter_expression: ds.Expression,
) -> tuple[ds.Fragment, ...] | None:
    get_fragments = getattr(dataset, "get_fragments", None)
    if not callable(get_fragments):
        return None
    try:
        fragments = get_fragments(filter=filter_expression)
        if not isinstance(fragments, Iterable):
            return None
        return tuple(fragments)
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None


def _string_method(
    field: ds.Expression,
    *,
    op: str,
) -> Callable[[str], ds.Expression] | None:
    if op == "contains":
        contains = getattr(field, "contains", None)
        return contains if callable(contains) else None
    if op == "startswith":
        starts_with = getattr(field, "starts_with", None)
        if callable(starts_with):
            return starts_with
        startswith = getattr(field, "startswith", None)
        return startswith if callable(startswith) else None
    return None


_DATASET_COMPARISON_BUILDERS: dict[str, Callable[[ds.Expression, FilterValue], ds.Expression]] = {
    "eq": lambda field, value: field == value,
    "ne": lambda field, value: field != value,
    "lt": lambda field, value: field < value,
    "lte": lambda field, value: field <= value,
    "gt": lambda field, value: field > value,
    "gte": lambda field, value: field >= value,
}


__all__ = [
    "DatasetManifestEntry",
    "DatasetManifestIndex",
    "dataset_filter_expression",
    "dataset_for_entry",
    "dataset_scanner_for_entry",
    "load_dataset_manifests",
]
