"""Parquet-backed safe query helpers for dataset snapshots."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.config.primitives import SnapshotRef
from codeintel.core.columnar.compute import (
    count_distinct,
    count_non_positive,
    count_true,
    orphan_ref_count,
)
from codeintel.core.columnar.compute_config import (
    DEFAULT_SCALAR_AGG,
    DEFAULT_SCALAR_AGG_ALLOW_NULL,
)
from codeintel.core.columnar.compute_helpers import call_compute
from codeintel.core.columnar.explode_ops import ExplodeSpec, explode_edges
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.iter import iter_array_values
from codeintel.core.columnar.masks import (
    filter_valid,
    is_valid_mask,
)
from codeintel.core.columnar.plan_ops import build_scan_plan
from codeintel.core.columnar.streaming import DatasetScanOptions
from codeintel.core.datasets.arrow_store import dataset_stats, scan_dataset
from codeintel.core.datasets.paths import dataset_snapshot_dir
from codeintel.core.datasets.scanner_ops import build_scanner
from codeintel.core.datasets.scanning import ParquetScanOptions, scan_parquet_table
from codeintel.core.query_results import ScalarCoercionError
from codeintel.core.table_key import is_valid_table_key
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
    filter_expr = _scope_filter_expression(
        dataset,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    count = _dataset_row_count(dataset, filter_expr=filter_expr)
    if count is not None:
        return count
    options = ParquetScanOptions(
        repo=snapshot.repo,
        commit=snapshot.commit,
        implicit_ordering=True,
        require_sequenced_output=True,
        metrics_enabled=True,
        finalize_mode="tolerant",
    )
    table = scan_parquet_table(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot.commit,
        options=options,
    )
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
    value = call_compute(
        "min",
        [table.column(column)],
        options=DEFAULT_SCALAR_AGG,
    )
    if not isinstance(value, pa.Scalar):
        return None
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
    value = call_compute(
        "max",
        [table.column(column)],
        options=DEFAULT_SCALAR_AGG,
    )
    if not isinstance(value, pa.Scalar):
        return None
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
    dataset = _open_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    if dataset is None or column not in dataset.schema.names:
        return 0
    filtered_count = _count_non_positive_filtered(dataset, column=column)
    if filtered_count is not None:
        return filtered_count
    table = _read_table(dataset, columns=[column])
    if table is None:
        return 0
    values = table.column(column)
    computed = _count_non_positive(values)
    if computed is not None:
        return computed
    count = sum(1 for value in iter_array_values(values) if _is_non_positive(value))
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
    total = 0
    distinct_values: set[object] = set()
    distinct_repr: set[str] = set()
    use_repr = False
    for value in iter_array_values(values):
        if value is None:
            continue
        total += 1
        if use_repr:
            distinct_repr.add(repr(value))
            continue
        try:
            distinct_values.add(value)
        except TypeError:
            use_repr = True
            distinct_repr = {repr(item) for item in distinct_values}
            distinct_values.clear()
            distinct_repr.add(repr(value))
    distinct = len(distinct_repr) if use_repr else len(distinct_values)
    return total - distinct


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
    total_value = call_compute(
        "count",
        [values],
        options=DEFAULT_SCALAR_AGG_ALLOW_NULL,
    )
    non_null_value = call_compute(
        "count",
        [values],
        options=DEFAULT_SCALAR_AGG,
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
    target_set: set[object] = set()
    for value in iter_array_values(target_values):
        if value is not None:
            target_set.add(value)
    count = 0
    for value in iter_array_values(source_values):
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


def _dataset_row_count(
    dataset: ds.Dataset,
    *,
    filter_expr: ds.Expression | None = None,
) -> int | None:
    if filter_expr is None:
        stats = dataset_stats(dataset)
        if stats.row_count is not None:
            return stats.row_count
    counter = getattr(dataset, "count_rows", None)
    if callable(counter):
        try:
            if filter_expr is None:
                return int(counter())
            return int(counter(filter=filter_expr))
        except (pa.ArrowInvalid, pa.ArrowTypeError, OSError, ValueError):
            pass
    try:
        plan = build_scan_plan(
            dataset,
            columns=None,
            filter_expr=filter_expr,
            implicit_ordering=True,
            require_sequenced_output=True,
        )
        counted = plan.aggregate(
            keys=[],
            aggregates=[(E.scalar(1), "count", None, "row_count")],
        )
        table = counted.to_table(use_threads=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return None
    if table.num_rows == 0 or "row_count" not in table.column_names:
        return None
    value = table.column("row_count")[0]
    if isinstance(value, pa.Scalar):
        return _as_int(value)
    return _as_int(value)


def _scope_filter_expression(
    dataset: ds.Dataset,
    *,
    repo: str,
    commit: str,
) -> ds.Expression | None:
    names = set(dataset.schema.names)
    expression: ds.Expression | None = None
    if "repo" in names:
        expression = ds.field("repo") == repo
    if "commit" in names:
        commit_expr = ds.field("commit") == commit
        expression = commit_expr if expression is None else expression & commit_expr
    return expression


def _read_table(
    dataset: ds.Dataset,
    *,
    columns: Iterable[str] | None = None,
    filter_expr: ds.Expression | None = None,
) -> pa.Table | None:
    try:
        plan = build_scan_plan(
            dataset,
            columns=list(columns) if columns is not None else None,
            filter_expr=filter_expr,
            implicit_ordering=True,
            require_sequenced_output=True,
        )
        return plan.to_table(use_threads=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        pass
    try:
        options = DatasetScanOptions(
            columns=list(columns) if columns is not None else None,
            filter_expression=filter_expr,
            implicit_ordering=True,
            require_sequenced_output=True,
            unify_schemas=True,
        )
        scanner = build_scanner(dataset, options=options)
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
    options = ParquetScanOptions(
        columns=[column],
        implicit_ordering=True,
        require_sequenced_output=True,
        metrics_enabled=True,
    )
    return scan_parquet_table(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=options,
    )


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


def _null_count(values: pa.ChunkedArray) -> int:
    mask = call_compute("is_null", [values])
    if not isinstance(mask, (pa.Array, pa.ChunkedArray)):
        return _as_int(values.null_count)
    return _sum_mask(mask)


def _sum_mask(mask: pa.Array | pa.ChunkedArray) -> int:
    total = call_compute(
        "sum",
        [mask],
        options=DEFAULT_SCALAR_AGG,
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


def _is_list_type(data_type: pa.DataType) -> bool:
    return bool(
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    )


def _flatten_list_values(values: pa.ChunkedArray) -> pa.ChunkedArray:
    if not _is_list_type(values.type):
        return values
    if len(values) == 0:
        return values
    row_ids = pa.arange(0, len(values))
    table = pa.table(
        {
            "__row_id": row_ids,
            "__values": values,
        }
    )
    spec = ExplodeSpec(
        src_col="__row_id",
        dst_list_col="__values",
        null_list_policy="empty",
        null_child_policy="drop",
        enforce_parent_valid=False,
    )
    try:
        exploded = explode_edges(table, spec=spec)
    except (RuntimeError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
        return values
    return cast("pa.ChunkedArray", exploded.good.column("__values"))


def _count_non_positive(values: pa.ChunkedArray) -> int | None:
    values = _flatten_list_values(values)
    data_type = values.type
    if not _is_numeric_type(data_type) or pa.types.is_dictionary(data_type):
        return None
    try:
        return count_non_positive(values)
    except (TypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
        return None


def _count_duplicates(values: pa.ChunkedArray) -> int | None:
    try:
        values = _flatten_list_values(values)
        valid_mask = is_valid_mask(values)
        total_count = count_true(valid_mask)
        filtered = filter_valid(values)
        distinct_count = count_distinct(filtered)
    except (TypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
        return None
    if distinct_count > total_count:
        return None
    return total_count - distinct_count


def _count_orphan_refs(
    *,
    source_values: pa.ChunkedArray,
    target_values: pa.ChunkedArray,
    allow_null: bool,
) -> int | None:
    try:
        source_values = _flatten_list_values(source_values)
        target_values = _flatten_list_values(target_values)
        filtered_target = filter_valid(target_values)
        return orphan_ref_count(
            source_values,
            filtered_target,
            allow_null=allow_null,
        )
    except (TypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
        return None


def _count_non_positive_filtered(dataset: ds.Dataset, *, column: str) -> int | None:
    if column not in dataset.schema.names:
        return None
    try:
        field = dataset.schema.field(column)
    except KeyError:
        return None
    if not _is_numeric_type(field.type) or pa.types.is_dictionary(field.type):
        return None
    filter_expr = ds.field(column) <= 0
    result: int | None = None
    try:
        plan = build_scan_plan(
            dataset,
            columns=[column],
            filter_expr=filter_expr,
            implicit_ordering=True,
            require_sequenced_output=True,
        )
        result = plan.to_table(use_threads=True).num_rows
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        result = None
    if result is None:
        try:
            options = DatasetScanOptions(
                columns=[column],
                filter_expression=filter_expr,
                implicit_ordering=True,
                require_sequenced_output=True,
                unify_schemas=True,
            )
            scanner = build_scanner(dataset, options=options)
            counter = getattr(scanner, "count_rows", None)
            result = _as_int(counter()) if callable(counter) else scanner.to_table().num_rows
        except (pa.ArrowInvalid, pa.ArrowTypeError, OSError, ValueError, TypeError):
            result = None
    return result


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
