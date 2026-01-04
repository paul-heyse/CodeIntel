"""Parquet-backed safe query helpers for dataset snapshots."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from codeintel.config.primitives import SnapshotRef
from codeintel.core.query_results import ScalarCoercionError
from codeintel.core.table_key import is_valid_table_key
from codeintel.storage.datasets.arrow_store import dataset_stats, scan_dataset
from codeintel.storage.datasets.paths import dataset_snapshot_dir
from codeintel.storage.query_results import coerce_optional_float

if TYPE_CHECKING:
    from collections.abc import Iterable

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ForeignKeyRef:
    """Foreign key reference specification for orphan counting."""

    source_table: str
    source_column: str
    ref_table: str
    ref_column: str
    allow_null: bool = True


def safe_table_exists(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> bool:
    """Return True when the snapshot dataset exists.

    Returns
    -------
    bool
        True when the dataset snapshot exists.
    """
    if not is_valid_table_key(table_key):
        return False
    return dataset_snapshot_dir(
        dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    ).is_dir()


def safe_get_columns(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> set[str]:
    """Return column names for a snapshot dataset.

    Returns
    -------
    set[str]
        Column names for the dataset snapshot.
    """
    dataset = _open_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    if dataset is None:
        return set()
    return set(dataset.schema.names)


def safe_count(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> int | None:
    """Safely count rows in a snapshot dataset.

    Returns
    -------
    int | None
        Row count or None when the dataset is missing.
    """
    dataset = _open_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    if dataset is None:
        return None
    count = _dataset_row_count(dataset)
    return count if count is not None else 0


def safe_count_with_scope(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot: SnapshotRef,
) -> int | None:
    """Safely count rows in a snapshot dataset scoped to repo/commit columns.

    Returns
    -------
    int | None
        Row count or None when the dataset is missing.
    """
    dataset = _open_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot.commit,
    )
    if dataset is None:
        return None
    filter_expr = _snapshot_filter(dataset.schema, snapshot=snapshot)
    table = _read_table(dataset, filter_expr=filter_expr)
    if table is None:
        return None
    return table.num_rows


def safe_count_nulls(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    column: str,
) -> int:
    """Count NULL values in a column.

    Returns
    -------
    int
        Number of NULL values (0 when missing).
    """
    table = _table_for_column(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        column=column,
    )
    if table is None:
        return 0
    values = table.column(column)
    return _null_count(values)


def safe_min_value(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    column: str,
) -> float | None:
    """Get minimum numeric value from a column.

    Returns
    -------
    float | None
        Minimum value, or None when empty.
    """
    table = _table_for_column(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        column=column,
    )
    if table is None or table.num_rows == 0:
        return None
    value = _compute_scalar(
        "min",
        [table.column(column)],
        options=pc.ScalarAggregateOptions(skip_nulls=True),
    )
    return _coerce_optional_float(value, ctx=f"{table_key}.{column}.min")


def safe_max_value(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    column: str,
) -> float | None:
    """Get maximum numeric value from a column.

    Returns
    -------
    float | None
        Maximum value, or None when empty.
    """
    table = _table_for_column(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        column=column,
    )
    if table is None or table.num_rows == 0:
        return None
    value = _compute_scalar(
        "max",
        [table.column(column)],
        options=pc.ScalarAggregateOptions(skip_nulls=True),
    )
    return _coerce_optional_float(value, ctx=f"{table_key}.{column}.max")


def safe_count_non_positive(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    column: str,
) -> int:
    """Count non-positive values in a numeric column.

    Returns
    -------
    int
        Count of values less than or equal to zero.
    """
    table = _table_for_column(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        column=column,
    )
    if table is None:
        return 0
    values = table.column(column)
    computed = _count_non_positive(values)
    if computed is not None:
        return computed
    count = sum(1 for value in values.to_pylist() if _is_non_positive(value))
    return int(count)


def safe_count_duplicates(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    column: str,
) -> int:
    """Count duplicate values in a column.

    Returns
    -------
    int
        Count of duplicate values.
    """
    table = _table_for_column(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        column=column,
    )
    if table is None:
        return 0
    values = table.column(column)
    computed = _count_duplicates(values)
    if computed is not None:
        return computed
    python_values = [value for value in values.to_pylist() if value is not None]
    try:
        distinct = len(set(python_values))
    except TypeError:
        distinct = len({repr(value) for value in python_values})
    return len(python_values) - distinct


def safe_not_null_fraction(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    column: str,
) -> float:
    """Get fraction of non-null values in a column.

    Returns
    -------
    float
        Fraction of non-null values (0.0 to 1.0).
    """
    table = _table_for_column(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        column=column,
    )
    if table is None:
        return 0.0
    total = table.num_rows
    if total == 0:
        return 0.0
    values = table.column(column)
    total_value = _compute_scalar(
        "count",
        [values],
        options=pc.ScalarAggregateOptions(skip_nulls=False),
    )
    non_null_value = _compute_scalar(
        "count",
        [values],
        options=pc.ScalarAggregateOptions(skip_nulls=True),
    )
    total_count = _as_int(total_value)
    non_null_count = _as_int(non_null_value)
    if total_count == 0:
        return 0.0
    return float(non_null_count) / float(total_count)


def safe_count_orphan_refs(
    *,
    dataset_root: Path,
    fk: ForeignKeyRef,
    snapshot_id: str,
) -> int:
    """Count orphaned foreign key references.

    Returns
    -------
    int
        Count of orphaned references.
    """
    source = _table_for_column(
        dataset_root=dataset_root,
        table_key=fk.source_table,
        snapshot_id=snapshot_id,
        column=fk.source_column,
    )
    target = _table_for_column(
        dataset_root=dataset_root,
        table_key=fk.ref_table,
        snapshot_id=snapshot_id,
        column=fk.ref_column,
    )
    if source is None or target is None:
        return 0
    source_values = source.column(fk.source_column)
    target_values = target.column(fk.ref_column)
    computed = _count_orphan_refs(
        source_values=source_values,
        target_values=target_values,
        allow_null=fk.allow_null,
    )
    if computed is not None:
        return computed
    target_filtered = [value for value in target_values.to_pylist() if value is not None]
    target_set = set(target_filtered)
    count = 0
    for value in source_values.to_pylist():
        if value is None:
            if fk.allow_null:
                count += 1
            continue
        if value not in target_set:
            count += 1
    return count


def _open_dataset(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> ds.Dataset | None:
    if not is_valid_table_key(table_key):
        return None
    try:
        return scan_dataset(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
    except (FileNotFoundError, OSError, ValueError, pa.ArrowInvalid) as exc:
        LOG.debug("Dataset scan failed for %s: %s", table_key, exc)
        return None


def _dataset_row_count(dataset: ds.Dataset) -> int | None:
    stats = dataset_stats(dataset)
    if stats.row_count is not None:
        return stats.row_count
    try:
        return dataset.to_table().num_rows
    except (pa.ArrowInvalid, pa.ArrowTypeError, OSError, ValueError):
        return None


def _snapshot_filter(schema: pa.Schema, *, snapshot: SnapshotRef) -> ds.Expression | None:
    columns = set(schema.names)
    if "repo" not in columns or "commit" not in columns:
        return None
    return (ds.field("repo") == snapshot.repo) & (ds.field("commit") == snapshot.commit)


def _read_table(
    dataset: ds.Dataset,
    *,
    columns: Iterable[str] | None = None,
    filter_expr: ds.Expression | None = None,
) -> pa.Table | None:
    try:
        scanner = dataset.scanner(
            columns=list(columns) if columns is not None else None,
            filter=filter_expr,
        )
        return scanner.to_table()
    except (pa.ArrowInvalid, pa.ArrowTypeError, OSError, ValueError):
        return None


def _table_for_column(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    column: str,
) -> pa.Table | None:
    dataset = _open_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    if dataset is None or column not in dataset.schema.names:
        return None
    return _read_table(dataset, columns=[column])


def _as_int(value: object | None) -> int:
    if value is None:
        return 0
    raw = cast("pa.Scalar", value).as_py() if isinstance(value, pa.Scalar) else value
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    return 0


def _coerce_optional_float(value: object | None, *, ctx: str) -> float | None:
    try:
        return coerce_optional_float(value, ctx=ctx)
    except ScalarCoercionError:
        return None


def _compute_scalar(
    name: str,
    args: list[object],
    *,
    options: pc.FunctionOptions | None = None,
) -> object | None:
    try:
        result = pc.call_function(name, args, options=options)
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError):
        return None
    return result.as_py() if hasattr(result, "as_py") else result


def _count_options_only_valid() -> pc.FunctionOptions | None:
    options_type = getattr(pc, "CountOptions", None)
    if options_type is None:
        return None
    try:
        return options_type(mode="only_valid")
    except TypeError:
        return None


def _compute_array(
    name: str,
    args: list[object],
) -> pa.Array | pa.ChunkedArray | None:
    try:
        result = pc.call_function(name, args)
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError):
        return None
    if isinstance(result, (pa.Array, pa.ChunkedArray)):
        return result
    return None


def _null_count(values: pa.ChunkedArray) -> int:
    mask = _compute_array("is_null", [values])
    if mask is None:
        return _as_int(values.null_count)
    return _sum_mask(mask)


def _sum_mask(mask: pa.Array | pa.ChunkedArray) -> int:
    total = _compute_scalar(
        "sum",
        [mask],
        options=pc.ScalarAggregateOptions(skip_nulls=True),
    )
    return _as_int(total)


def _is_non_positive(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value <= 0
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        try:
            return float(stripped) <= 0
        except ValueError:
            return False
    return False


def _is_numeric_type(data_type: pa.DataType) -> bool:
    if pa.types.is_boolean(data_type):
        return False
    return bool(
        pa.types.is_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_decimal(data_type)
    )


def _numeric_zero(data_type: pa.DataType) -> pa.Scalar | None:
    try:
        return pa.scalar(0, type=data_type)
    except (TypeError, pa.ArrowInvalid):
        return None


def _count_non_positive(values: pa.ChunkedArray) -> int | None:
    data_type = values.type
    if not _is_numeric_type(data_type) or pa.types.is_dictionary(data_type):
        return None
    scalar = _numeric_zero(data_type)
    if scalar is None:
        return None
    mask = _compute_array("less_equal", [values, scalar])
    if mask is None:
        return None
    return _sum_mask(mask)


def _count_duplicates(values: pa.ChunkedArray) -> int | None:
    options = _count_options_only_valid()
    total_value = _compute_scalar("count", [values], options=options)
    if total_value is None and options is not None:
        total_value = _compute_scalar("count", [values])
    if total_value is None:
        return None
    distinct_value = _compute_scalar("count_distinct", [values], options=options)
    if distinct_value is None and options is not None:
        distinct_value = _compute_scalar("count_distinct", [values])
    if distinct_value is None:
        return None
    total_count = _as_int(total_value)
    distinct_count = _as_int(distinct_value)
    if distinct_count > total_count:
        return None
    return total_count - distinct_count


def _count_orphan_refs(
    *,
    source_values: pa.ChunkedArray,
    target_values: pa.ChunkedArray,
    allow_null: bool,
) -> int | None:
    filtered_target = _filter_valid_values(target_values)
    in_mask = (
        _compute_array("is_in", [source_values, filtered_target])
        if filtered_target is not None
        else None
    )
    not_in_mask = _compute_array("invert", [in_mask]) if in_mask is not None else None
    valid_mask = _compute_array("is_valid", [source_values]) if not_in_mask is not None else None
    orphan_mask = (
        _compute_array("and_kleene", [valid_mask, not_in_mask])
        if valid_mask is not None
        else None
    )
    orphan_count = _sum_mask(orphan_mask)
    if allow_null:
        null_mask = _compute_array("is_null", [source_values])
        if null_mask is None:
            return None
        orphan_count += _sum_mask(null_mask)
    return orphan_count


def _filter_valid_values(
    values: pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray | None:
    mask = _compute_array("is_valid", [values])
    if mask is None:
        return None
    return _compute_array("filter", [values, mask])


__all__ = [
    "ForeignKeyRef",
    "safe_count",
    "safe_count_duplicates",
    "safe_count_non_positive",
    "safe_count_nulls",
    "safe_count_orphan_refs",
    "safe_count_with_scope",
    "safe_get_columns",
    "safe_max_value",
    "safe_min_value",
    "safe_not_null_fraction",
    "safe_table_exists",
]
