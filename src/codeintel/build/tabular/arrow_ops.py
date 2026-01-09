"""Arrow-first join and materialization helpers for build pipelines.

Policy
------
- Graph compute modules use Arrow tables/readers end-to-end and call these helpers.
- Polars fallbacks are reserved for legacy or view/export paths only.
- Join keys and cardinality expectations live in `docs/architecture/arrow_join_policy.md`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict, Unpack, cast

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as paj

from codeintel.build.contracts.registry import (
    ContractedTableContext,
    require_contract_for_target,
)
from codeintel.build.contracts.types import ContractPolicy
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular import array_ops as _array_ops
from codeintel.build.tabular.compute_helpers import (
    array_from_compute,
    cast_array,
    scalar_from_compute,
)
from codeintel.build.tabular.conversion import (
    arrow_reader_to_lazyframe,
    lazyframe_to_reader,
    reader_to_table,
    table_to_frame,
    table_to_reader,
    tabular_to_arrow_reader,
)
from codeintel.build.tabular.dedupe_ops import dedupe_table_for_table, dedupe_tabular
from codeintel.build.tabular.frames import (
    JoinSpec,
    JoinStrategy,
    JoinValidation,
    join_validated,
)
from codeintel.build.tabular.plan_ops import HashJoinSpec, Plan
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.iter import (
    iter_array_values as _iter_array_values,
)
from codeintel.core.columnar.iter import (
    iter_rows as _iter_rows,
)
from codeintel.core.columnar.normalization import (
    normalize_table_for_compute as _normalize_table_for_compute,
)
from codeintel.core.columnar.schema_alignment import align_reader_to_contract as _align_reader
from codeintel.core.columnar.schema_metadata import decode_metadata
from codeintel.core.columnar.schema_ops import concat_tables_unified as _concat_tables_unified
from codeintel.core.columnar.streaming import configure_arrow_threading
from codeintel.core.constants import (
    DEFAULT_ARROW_BATCH_READAHEAD,
    DEFAULT_ARROW_BATCH_SIZE,
    DEFAULT_ARROW_CACHE_METADATA,
    DEFAULT_ARROW_FRAGMENT_READAHEAD,
    DEFAULT_ARROW_PARQUET_BUFFER_SIZE,
    DEFAULT_ARROW_PARQUET_PRE_BUFFER,
    DEFAULT_ARROW_PARQUET_USE_BUFFERED_STREAM,
    DEFAULT_ARROW_USE_THREADS,
)
from codeintel.core.datasets.scanning import (
    ParquetScanOptions,
    scan_parquet_dataset,
    scan_parquet_table,
)
from codeintel.core.schemas.arrow_gen import (
    DEFAULT_EXTRAS_COLUMN,
    ArrowSchemaMetadata,
    ExtrasPolicy,
    arrow_contract_for_table_schema,
)

ensure_array = _array_ops.ensure_array
index_in = _array_ops.index_in
normalize_binary_view_array = _array_ops.normalize_binary_view_array
normalize_string_view_array = _array_ops.normalize_string_view_array
take_by_key = _array_ops.take_by_key
value_set_array = _array_ops.value_set_array

_ARROW_JOIN_TYPES = {
    "left": "left outer",
    "inner": "inner",
    "right": "right outer",
    "full": "full outer",
    "outer": "full outer",
}

LOG = logging.getLogger(__name__)
_JOIN_THREAD_THRESHOLD = 200_000


@dataclass(frozen=True, slots=True)
class ArrowJoinSpec:
    """Arrow join configuration for materialized joins."""

    on: Sequence[str] | None = None
    left_on: Sequence[str] | None = None
    right_on: Sequence[str] | None = None
    how: JoinStrategy = "left"
    validate: JoinValidation | None = None
    suffix: str = ""
    coalesce_keys: bool = True
    left_suffix: str | None = None
    right_suffix: str | None = None


@dataclass(frozen=True, slots=True)
class ArrowJoinOptions:
    """Optional Arrow join tuning for filters and normalization."""

    filter_expression: pc.Expression | None = None
    use_threads: bool | None = True
    normalize_inputs: bool = True


JoinFilterSide = Literal["left", "right", "either"]


@dataclass(frozen=True, slots=True)
class JoinFilterClause:
    """Specification for a residual join filter."""

    field: str
    predicate: Callable[[str], pc.Expression]
    side: JoinFilterSide = "either"


@dataclass(frozen=True, slots=True)
class ParquetScanSpec:
    """Parquet scan settings for snapshot retrieval."""

    dataset_root: Path
    table_key: str
    snapshot_id: str
    columns: Sequence[str] | None = None
    repo: str | None = None
    commit: str | None = None
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE
    batch_readahead: int | None = DEFAULT_ARROW_BATCH_READAHEAD
    fragment_readahead: int | None = DEFAULT_ARROW_FRAGMENT_READAHEAD
    use_threads: bool | None = DEFAULT_ARROW_USE_THREADS
    cache_metadata: bool | None = DEFAULT_ARROW_CACHE_METADATA
    parquet_pre_buffer: bool | None = DEFAULT_ARROW_PARQUET_PRE_BUFFER
    parquet_use_buffered_stream: bool | None = DEFAULT_ARROW_PARQUET_USE_BUFFERED_STREAM
    parquet_buffer_size: int | None = DEFAULT_ARROW_PARQUET_BUFFER_SIZE


def _arrow_spec_from_join_spec(spec: JoinSpec) -> ArrowJoinSpec:
    """Create an ArrowJoinSpec from a JoinSpec.

    Returns
    -------
    ArrowJoinSpec
        Arrow join configuration.
    """
    return ArrowJoinSpec(
        on=spec.on,
        left_on=spec.left_on,
        right_on=spec.right_on,
        how=spec.how,
        validate=spec.validate,
        suffix=spec.suffix,
    )


def _join_spec_from_arrow_spec(spec: ArrowJoinSpec) -> JoinSpec:
    """Build a JoinSpec fallback from an ArrowJoinSpec.

    Returns
    -------
    JoinSpec
        Join configuration for LazyFrame joins.
    """
    suffix = spec.suffix
    if not suffix and spec.right_suffix:
        suffix = spec.right_suffix
    return JoinSpec(
        on=spec.on,
        left_on=spec.left_on,
        right_on=spec.right_on,
        how=spec.how,
        validate=spec.validate,
        suffix=suffix,
    )


def _polars_join_fallback(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    spec: ArrowJoinSpec | JoinSpec,
) -> pl.DataFrame:
    join_spec = _join_spec_from_arrow_spec(spec) if isinstance(spec, ArrowJoinSpec) else spec
    left_lazy = left.lazy() if isinstance(left, pl.DataFrame) else left
    right_lazy = right.lazy() if isinstance(right, pl.DataFrame) else right
    left_lazy, right_lazy = _coerce_null_join_keys(left_lazy, right_lazy, spec=join_spec)
    joined = join_validated(left_lazy, right_lazy, spec=join_spec)
    joined = _ensure_right_join_keys(joined, spec=join_spec)
    return joined.collect()


def _coerce_null_join_keys(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    *,
    spec: JoinSpec,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    left_keys, right_keys = _resolve_join_keys(
        on=spec.on,
        left_on=spec.left_on,
        right_on=spec.right_on,
    )
    if not left_keys:
        return left, right
    resolved_right = right_keys if right_keys is not None else left_keys
    if len(left_keys) != len(resolved_right):
        return left, right
    left_schema = left.collect_schema()
    right_schema = right.collect_schema()
    for left_key, right_key in zip(left_keys, resolved_right, strict=True):
        left_dtype = left_schema.get(left_key)
        right_dtype = right_schema.get(right_key)
        if left_dtype is None or right_dtype is None:
            continue
        if left_dtype == pl.Null and right_dtype != pl.Null:
            left = left.with_columns(pl.col(left_key).cast(right_dtype))
        elif right_dtype == pl.Null and left_dtype != pl.Null:
            right = right.with_columns(pl.col(right_key).cast(left_dtype))
    return left, right


def _ensure_right_join_keys(
    frame: pl.LazyFrame,
    *,
    spec: JoinSpec,
) -> pl.LazyFrame:
    left_keys, right_keys = _resolve_join_keys(
        on=spec.on,
        left_on=spec.left_on,
        right_on=spec.right_on,
    )
    if not right_keys:
        return frame
    resolved_right = right_keys
    if left_keys == resolved_right:
        return frame
    schema_names = set(frame.collect_schema().names())
    expressions: list[pl.Expr] = []
    for left_key, right_key in zip(left_keys, resolved_right, strict=True):
        if right_key in schema_names or left_key == right_key:
            continue
        expressions.append(pl.col(left_key).alias(right_key))
    if not expressions:
        return frame
    return frame.with_columns(expressions)


def _resolve_join_keys(
    *,
    on: Sequence[str] | None,
    left_on: Sequence[str] | None,
    right_on: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    if on:
        return tuple(on), tuple(right_on) if right_on else None
    if left_on:
        return tuple(left_on), tuple(right_on) if right_on else None
    if right_on:
        return tuple(right_on), tuple(right_on)
    return (), None


def _coalesced_keys(
    keys: Sequence[str],
    right_keys: Sequence[str] | None,
    *,
    coalesce_keys: bool,
) -> set[str]:
    if not coalesce_keys:
        return set()
    resolved_right = keys if right_keys is None else right_keys
    return set(keys) & set(resolved_right)


def _resolved_right_suffix(
    *,
    left_columns: set[str],
    right_columns: set[str],
    keys: Sequence[str],
    right_keys: Sequence[str] | None,
    spec: ArrowJoinSpec,
) -> str | None:
    right_suffix = spec.right_suffix
    if spec.suffix and right_suffix is None:
        right_suffix = spec.suffix
    if right_suffix in {None, ""}:
        overlapping = left_columns & right_columns
        coalesced = _coalesced_keys(keys, right_keys, coalesce_keys=spec.coalesce_keys)
        if overlapping - coalesced:
            right_suffix = "_right"
    return right_suffix


def resolve_join_filter_field(
    field: str,
    *,
    left: pa.Table,
    right: pa.Table,
    spec: ArrowJoinSpec,
    side: JoinFilterSide = "either",
) -> str | None:
    """Resolve a join output field name for filter expressions.

    Returns
    -------
    str | None
        Resolved field name for the join output, if present.
    """
    left_columns = set(left.column_names)
    right_columns = set(right.column_names)
    keys, right_keys = _resolve_join_keys(on=spec.on, left_on=spec.left_on, right_on=spec.right_on)
    coalesced = _coalesced_keys(keys, right_keys, coalesce_keys=spec.coalesce_keys)
    right_suffix = _resolved_right_suffix(
        left_columns=left_columns,
        right_columns=right_columns,
        keys=keys,
        right_keys=right_keys,
        spec=spec,
    )
    if side in {"left", "either"} and field in left_columns:
        if spec.left_suffix and field in right_columns and field not in coalesced:
            return f"{field}{spec.left_suffix}"
        return field
    if side in {"right", "either"} and field in right_columns:
        if field in coalesced:
            return None
        if field in left_columns and right_suffix not in {None, ""}:
            return f"{field}{right_suffix}"
        return field
    return None


def join_filter_expr(
    *,
    left: pa.Table,
    right: pa.Table,
    spec: ArrowJoinSpec,
    clause: JoinFilterClause,
) -> pc.Expression | None:
    """Build a join filter expression if the field exists post-join.

    Returns
    -------
    pyarrow.compute.Expression | None
        Join filter expression when the field exists.
    """
    resolved = resolve_join_filter_field(
        clause.field,
        left=left,
        right=right,
        spec=spec,
        side=clause.side,
    )
    if resolved is None:
        return None
    return clause.predicate(resolved)


def combine_join_filters(*expressions: pc.Expression | None) -> pc.Expression | None:
    """Combine join filter expressions using AND semantics.

    Returns
    -------
    pyarrow.compute.Expression | None
        Combined expression when inputs are provided.
    """
    resolved = [expr for expr in expressions if expr is not None]
    if not resolved:
        return None
    combined = resolved[0]
    for expr in resolved[1:]:
        combined &= expr
    return combined


def build_join_options(
    left: pa.Table,
    right: pa.Table,
    *,
    filter_expression: pc.Expression | None = None,
    use_threads: bool | None = None,
    normalize_inputs: bool = True,
) -> ArrowJoinOptions:
    """Build join options with a size-based threading heuristic.

    Returns
    -------
    ArrowJoinOptions
        Join options for Arrow joins.
    """
    resolved_threads = use_threads
    if resolved_threads is None:
        total_rows = left.num_rows + right.num_rows
        resolved_threads = total_rows >= _JOIN_THREAD_THRESHOLD
    return ArrowJoinOptions(
        filter_expression=filter_expression,
        use_threads=resolved_threads,
        normalize_inputs=normalize_inputs,
    )


def _is_list_type(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    )


def _string_view_cast_dictionary(data_type: pa.DataType) -> pa.DataType:
    value_type = _string_view_cast_type(data_type.value_type)
    if value_type == data_type.value_type:
        return data_type
    return pa.dictionary(data_type.index_type, value_type, ordered=data_type.ordered)


def _string_view_cast_list(data_type: pa.DataType) -> pa.DataType:
    value_type = _string_view_cast_type(data_type.value_type)
    if value_type == data_type.value_type:
        return data_type
    if pa.types.is_large_list(data_type):
        return pa.large_list(value_type)
    if pa.types.is_list(data_type):
        return pa.list_(value_type)
    return pa.list_(value_type, list_size=data_type.list_size)


def _string_view_cast_struct(data_type: pa.DataType) -> pa.DataType:
    fields: list[pa.Field] = []
    changed = False
    for field in data_type:
        next_type = _string_view_cast_type(field.type)
        if next_type != field.type:
            changed = True
        fields.append(
            pa.field(
                field.name,
                next_type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
        )
    if not changed:
        return data_type
    return pa.struct(fields)


def _string_view_cast_map(data_type: pa.DataType) -> pa.DataType:
    key_type = _string_view_cast_type(data_type.key_type)
    item_type = _string_view_cast_type(data_type.item_type)
    if key_type == data_type.key_type and item_type == data_type.item_type:
        return data_type
    return pa.map_(key_type, item_type, keys_sorted=data_type.keys_sorted)


def _string_view_cast_type(data_type: pa.DataType) -> pa.DataType:
    if pa.types.is_string_view(data_type):
        return pa.string()
    if pa.types.is_dictionary(data_type):
        return _string_view_cast_dictionary(data_type)
    if _is_list_type(data_type):
        return _string_view_cast_list(data_type)
    if pa.types.is_struct(data_type):
        return _string_view_cast_struct(data_type)
    if pa.types.is_map(data_type):
        return _string_view_cast_map(data_type)
    return data_type


def _is_binary_view_type(data_type: pa.DataType) -> bool:
    is_binary_view = getattr(pa.types, "is_binary_view", None)
    if is_binary_view is None:
        return False
    return is_binary_view(data_type)


def _binary_view_cast_dictionary(data_type: pa.DataType) -> pa.DataType:
    value_type = _binary_view_cast_type(data_type.value_type)
    if value_type == data_type.value_type:
        return data_type
    return pa.dictionary(data_type.index_type, value_type, ordered=data_type.ordered)


def _binary_view_cast_list(data_type: pa.DataType) -> pa.DataType:
    value_type = _binary_view_cast_type(data_type.value_type)
    if value_type == data_type.value_type:
        return data_type
    if pa.types.is_large_list(data_type):
        return pa.large_list(value_type)
    if pa.types.is_list(data_type):
        return pa.list_(value_type)
    return pa.list_(value_type, list_size=data_type.list_size)


def _binary_view_cast_struct(data_type: pa.DataType) -> pa.DataType:
    fields: list[pa.Field] = []
    changed = False
    for field in data_type:
        next_type = _binary_view_cast_type(field.type)
        if next_type != field.type:
            changed = True
        fields.append(
            pa.field(
                field.name,
                next_type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
        )
    if not changed:
        return data_type
    return pa.struct(fields)


def _binary_view_cast_map(data_type: pa.DataType) -> pa.DataType:
    key_type = _binary_view_cast_type(data_type.key_type)
    item_type = _binary_view_cast_type(data_type.item_type)
    if key_type == data_type.key_type and item_type == data_type.item_type:
        return data_type
    return pa.map_(key_type, item_type, keys_sorted=data_type.keys_sorted)


def _binary_view_cast_type(data_type: pa.DataType) -> pa.DataType:
    if _is_binary_view_type(data_type):
        return pa.binary()
    if pa.types.is_dictionary(data_type):
        return _binary_view_cast_dictionary(data_type)
    if _is_list_type(data_type):
        return _binary_view_cast_list(data_type)
    if pa.types.is_struct(data_type):
        return _binary_view_cast_struct(data_type)
    if pa.types.is_map(data_type):
        return _binary_view_cast_map(data_type)
    return data_type


def _cast_string_view_column(
    column: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    target_type = _string_view_cast_type(column.type)
    if target_type == column.type:
        return column
    try:
        return cast_array(column, target_type, safe=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
        return column


def _cast_binary_view_column(
    column: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    target_type = _binary_view_cast_type(column.type)
    if target_type == column.type:
        return column
    try:
        return cast_array(column, target_type, safe=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
        return column


def _group_by_table_keys(table: pa.Table, keys: Sequence[str]) -> pa.Table:
    columns = [_cast_string_view_column(table[key]) for key in keys]
    return pa.Table.from_arrays(columns, names=list(keys))


def _normalize_join_key_columns(table: pa.Table, keys: Sequence[str]) -> pa.Table:
    if not keys:
        return table
    columns: list[pa.Array | pa.ChunkedArray] = []
    changed = False
    key_set = set(keys)
    for name in table.column_names:
        column = table[name]
        if name in key_set:
            casted = _cast_string_view_column(column)
            if casted is not column:
                column = casted
                changed = True
        columns.append(column)
    if not changed:
        return table
    return pa.Table.from_arrays(columns, names=list(table.column_names))


def _normalize_table_string_views(table: pa.Table) -> pa.Table:
    columns: list[pa.Array | pa.ChunkedArray] = []
    changed = False
    for name in table.column_names:
        column = table[name]
        casted = _cast_string_view_column(column)
        if casted is not column:
            column = casted
            changed = True
        columns.append(column)
    if not changed:
        return table
    return pa.Table.from_arrays(columns, names=list(table.column_names))


def _normalize_table_binary_views(table: pa.Table) -> pa.Table:
    columns: list[pa.Array | pa.ChunkedArray] = []
    changed = False
    for name in table.column_names:
        column = table[name]
        casted = _cast_binary_view_column(column)
        if casted is not column:
            column = casted
            changed = True
        columns.append(column)
    if not changed:
        return table
    return pa.Table.from_arrays(columns, names=list(table.column_names))


def normalize_table_for_join(table: pa.Table, *, combine_chunks: bool = True) -> pa.Table:
    """Normalize string/binary view types ahead of Arrow joins.

    Returns
    -------
    pyarrow.Table
        Table with view types normalized, dictionaries unified, and chunks combined.
    """
    configure_arrow_threading()
    normalized = _normalize_table_binary_views(_normalize_table_string_views(table))
    return _normalize_table_for_compute(normalized, combine_chunks=combine_chunks)


def normalize_table_for_compute(table: pa.Table, *, combine_chunks: bool = True) -> pa.Table:
    """Normalize a table for compute-heavy kernels.

    Returns
    -------
    pa.Table
        Table with normalized view types, unified dictionaries, and combined chunks.
    """
    return normalize_table_for_join(table, combine_chunks=combine_chunks)


def _null_key_stats(
    table: pa.Table,
    keys: Sequence[str],
) -> tuple[bool, int | None]:
    null_mask: pa.Array | pa.ChunkedArray | None = None
    for key in keys:
        key_mask = array_from_compute("is_null", [table[key]])
        if key_mask is None:
            msg = "Arrow compute is_null did not return an array."
            raise TypeError(msg)
        if null_mask is None:
            null_mask = key_mask
        else:
            combined = array_from_compute("or_kleene", [null_mask, key_mask])
            if combined is None:
                msg = "Arrow compute or_kleene did not return an array."
                raise TypeError(msg)
            null_mask = combined
    if null_mask is None:
        return False, None
    any_null = scalar_from_compute("any", [null_mask])
    if not any_null:
        return False, None
    null_count = scalar_from_compute("sum", [pc.cast(null_mask, pa.int64())])
    if isinstance(null_count, int):
        return True, null_count
    return True, None


def _ensure_unique_keys(table: pa.Table, keys: Sequence[str], *, label: str) -> None:
    if not keys:
        return
    missing = [key for key in keys if key not in table.column_names]
    if missing:
        msg = f"Missing join keys on {label}: {', '.join(missing)}"
        raise ValueError(msg)
    has_nulls, null_count = _null_key_stats(table, keys)
    if has_nulls:
        count_info = f" (rows={null_count})" if isinstance(null_count, int) else ""
        msg = f"Join validation failed for {label}: NULL keys detected{count_info}"
        raise ValueError(msg)
    count_source = keys[0]
    grouped = (
        _group_by_table_keys(table, keys).group_by(list(keys)).aggregate([(count_source, "count")])
    )
    count_name = f"{count_source}_count"
    if grouped.num_rows == 0 or count_name not in grouped.column_names:
        return
    max_value = scalar_from_compute(
        "max",
        [grouped[count_name]],
        options=pc.ScalarAggregateOptions(skip_nulls=True),
    )
    if isinstance(max_value, (int, float)) and not isinstance(max_value, bool) and max_value > 1:
        msg = f"Join validation failed for {label}: keys not unique"
        raise ValueError(msg)
    if max_value is not None:
        return
    try:
        counts = grouped[count_name].combine_chunks().to_numpy(zero_copy_only=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return
    if counts.size > 0 and counts.max() > 1:
        msg = f"Join validation failed for {label}: keys not unique"
        raise ValueError(msg)


def _validate_join(
    left: pa.Table,
    right: pa.Table,
    *,
    left_keys: Sequence[str],
    right_keys: Sequence[str] | None,
    validate: JoinValidation | None,
) -> None:
    if validate is None or validate == "m:m":
        return
    right_key_values = right_keys if right_keys is not None else left_keys
    if validate in {"m:1", "1:1"}:
        _ensure_unique_keys(right, right_key_values, label="right")
    if validate in {"1:m", "1:1"}:
        _ensure_unique_keys(left, left_keys, label="left")
    if validate not in {"m:1", "1:m", "1:1", "m:m"}:
        msg = f"Unsupported join validation: {validate}"
        raise ValueError(msg)


def arrow_table_from_tabular(value: InferableTabularInput) -> pa.Table:
    """Convert a tabular input into a fully materialized Arrow Table.

    Returns
    -------
    pa.Table
        Materialized Arrow table.
    """
    reader = tabular_to_arrow_reader(value)
    return normalize_table_for_compute(reader_to_table(reader))


def arrow_table_from_lazyframe(frame: pl.LazyFrame) -> pa.Table:
    """Collect a LazyFrame into an Arrow Table.

    Returns
    -------
    pa.Table
        Materialized Arrow table.
    """
    return normalize_table_for_compute(reader_to_table(lazyframe_to_reader(frame)))


def _arrow_table_from_frame(frame: pl.DataFrame | pl.LazyFrame) -> pa.Table:
    if isinstance(frame, pl.DataFrame):
        return frame.to_arrow()
    return arrow_table_from_lazyframe(frame)


def arrow_join_tables(
    left: pa.Table,
    right: pa.Table,
    *,
    spec: ArrowJoinSpec,
    options: ArrowJoinOptions | None = None,
) -> pa.Table:
    """Join two Arrow tables using the provided keys.

    Parameters
    ----------
    left
        Left-hand table.
    right
        Right-hand table.
    spec
        Join configuration for Arrow joins.
    options
        Optional tuning for join filters, threading, and normalization.

    Raises
    ------
    ValueError
        If join keys are missing or if validation fails.
    TypeError
        If Arrow does not accept join options and no fallback applies.

    Returns
    -------
    pa.Table
        Joined Arrow table.
    """
    resolved_options = options or ArrowJoinOptions()
    filter_expression = resolved_options.filter_expression
    use_threads = resolved_options.use_threads
    normalize_inputs = resolved_options.normalize_inputs
    keys, right_keys = _resolve_join_keys(
        on=spec.on,
        left_on=spec.left_on,
        right_on=spec.right_on,
    )
    if not keys:
        msg = "Arrow join requires join keys"
        raise ValueError(msg)
    right_suffix = spec.right_suffix
    if spec.suffix and right_suffix is None:
        right_suffix = spec.suffix
    if right_suffix in {None, ""}:
        resolved_right_keys = keys if right_keys is None else right_keys
        overlapping = set(left.column_names) & set(right.column_names)
        coalesced = set(keys) & set(resolved_right_keys) if spec.coalesce_keys else set()
        if overlapping - coalesced:
            right_suffix = "_right"
    resolved_right_keys = keys if right_keys is None else right_keys
    if normalize_inputs:
        left = normalize_table_for_join(_normalize_join_key_columns(left, keys))
        right = normalize_table_for_join(_normalize_join_key_columns(right, resolved_right_keys))
    _validate_join(
        left,
        right,
        left_keys=keys,
        right_keys=right_keys,
        validate=spec.validate,
    )
    join_type = _ARROW_JOIN_TYPES.get(spec.how, spec.how)
    join_kwargs = {
        "keys": tuple(keys),
        "right_keys": tuple(right_keys) if right_keys is not None else None,
        "join_type": join_type,
        "left_suffix": spec.left_suffix,
        "right_suffix": right_suffix,
        "coalesce_keys": spec.coalesce_keys,
    }
    if filter_expression is not None:
        join_kwargs["filter_expression"] = filter_expression
    if use_threads is not None:
        join_kwargs["use_threads"] = use_threads
    try:
        return left.join(right, **join_kwargs)
    except TypeError:
        if filter_expression is None and use_threads is None:
            raise
        join_kwargs.pop("filter_expression", None)
        join_kwargs.pop("use_threads", None)
        return left.join(right, **join_kwargs)


@dataclass(frozen=True, slots=True)
class ContractAlignmentPlan:
    """Cached alignment plan for contract-aware schema alignment."""

    table_key: str
    target_name: str | None
    contract_schema: pa.Schema
    extras_policy: ExtrasPolicy | None


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """Diagnostics for contract alignment differences."""

    table_key: str
    target_name: str | None
    missing_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]
    coerced_columns: tuple[str, ...]
    row_count: int | None


AlignmentReporter = Callable[[AlignmentReport], None]
type AlignmentReportKey = tuple[str, str | None]


class AlignmentOverrides(TypedDict, total=False):
    """Keyword overrides for contract alignment helpers."""

    target_name: str | None
    policy: ContractPolicy | None
    extras_policy: ExtrasPolicy | None
    reporter: AlignmentReporter | None


@dataclass(frozen=True, slots=True)
class AlignmentOptions:
    """Options for contract alignment helpers."""

    target_name: str | None = None
    policy: ContractPolicy | None = None
    extras_policy: ExtrasPolicy | None = None
    reporter: AlignmentReporter | None = None


def _merge_alignment_options(
    options: AlignmentOptions | None,
    overrides: AlignmentOverrides,
) -> AlignmentOptions:
    resolved = options or AlignmentOptions()
    if overrides:
        return replace(resolved, **overrides)
    return resolved


def _alignment_options_from_context(
    context: ContractedTableContext,
    *,
    reporter: AlignmentReporter | None = None,
    extras_policy: ExtrasPolicy | None = None,
) -> AlignmentOptions:
    resolved_policy = context.policy or context.contract.policy
    return AlignmentOptions(
        target_name=context.contract.target,
        policy=resolved_policy,
        extras_policy=extras_policy,
        reporter=reporter,
    )


def _normalize_schema_for_report(schema: pa.Schema) -> pa.Schema:
    fields: list[pa.Field] = []
    changed = False
    for field in schema:
        normalized_type = _binary_view_cast_type(_string_view_cast_type(field.type))
        if normalized_type != field.type:
            changed = True
            normalized_field = field.with_type(normalized_type)
        else:
            normalized_field = field
        fields.append(normalized_field)
    if not changed:
        return schema
    return pa.schema(fields, metadata=schema.metadata)


def _extras_column_name(contract_schema: pa.Schema) -> str:
    metadata = decode_metadata(contract_schema.metadata)
    raw = metadata.get("codeintel.extras_column")
    if isinstance(raw, str) and raw:
        return raw
    return DEFAULT_EXTRAS_COLUMN


def _coerced_columns(
    contract_schema: pa.Schema,
    incoming_schema: pa.Schema,
) -> tuple[str, ...]:
    mismatched: list[str] = []
    incoming_names = set(incoming_schema.names)
    for field in contract_schema:
        if field.name not in incoming_names:
            continue
        incoming_field = incoming_schema.field(field.name)
        if incoming_field.type != field.type:
            mismatched.append(field.name)
    return tuple(sorted(mismatched))


def _build_alignment_report(
    *,
    table_key: str,
    target_name: str | None,
    contract_schema: pa.Schema,
    incoming_schema: pa.Schema,
    row_count: int | None,
) -> AlignmentReport:
    normalized_contract = _normalize_schema_for_report(contract_schema)
    normalized_incoming = _normalize_schema_for_report(incoming_schema)
    contract_names = {name for name in normalized_contract.names if isinstance(name, str)}
    incoming_names = {name for name in normalized_incoming.names if isinstance(name, str)}
    extras_column = _extras_column_name(normalized_contract)
    missing_columns = tuple(sorted(contract_names - incoming_names))
    extra_columns = tuple(
        name
        for name in sorted(incoming_names - contract_names)
        if isinstance(name, str) and name != extras_column
    )
    coerced_columns = _coerced_columns(normalized_contract, normalized_incoming)
    return AlignmentReport(
        table_key=table_key,
        target_name=target_name,
        missing_columns=missing_columns,
        extra_columns=extra_columns,
        coerced_columns=coerced_columns,
        row_count=row_count,
    )


_ALIGNMENT_REPORT_SEEN: set[AlignmentReportKey] = set()
_ALIGNMENT_REPORTS: dict[AlignmentReportKey, AlignmentReport] = {}
_ALIGNMENT_DIAGNOSTICS: dict[AlignmentReportKey, AlignmentReport] = {}


def record_alignment_report(report: AlignmentReport) -> None:
    """Store the latest alignment report for a table target."""
    _ALIGNMENT_REPORTS[report.table_key, report.target_name] = report


def record_alignment_diagnostic(report: AlignmentReport) -> None:
    """Store alignment diagnostics for contract drift persistence."""
    _ALIGNMENT_DIAGNOSTICS[report.table_key, report.target_name] = report


def drain_alignment_diagnostics() -> tuple[AlignmentReport, ...]:
    """Return and clear stored alignment diagnostics.

    Returns
    -------
    tuple[AlignmentReport, ...]
        Alignment diagnostics captured for persistence.
    """
    diagnostics = tuple(_ALIGNMENT_DIAGNOSTICS.values())
    _ALIGNMENT_DIAGNOSTICS.clear()
    return diagnostics


def pop_alignment_report(
    *,
    table_key: str,
    target_name: str | None,
) -> AlignmentReport | None:
    """Return and clear the latest alignment report for a table target.

    Returns
    -------
    AlignmentReport | None
        Report when available, otherwise None.
    """
    return _ALIGNMENT_REPORTS.pop((table_key, target_name), None)


def emit_alignment_report(report: AlignmentReport) -> None:
    """Log alignment diagnostics once per table target."""
    if not report.missing_columns and not report.extra_columns and not report.coerced_columns:
        return
    record_alignment_report(report)
    key = (report.table_key, report.target_name)
    if key in _ALIGNMENT_REPORT_SEEN:
        return
    _ALIGNMENT_REPORT_SEEN.add(key)
    LOG.warning(
        "build.contract_alignment table_key=%s target=%s missing=%s extra=%s coerced=%s "
        "row_count=%s",
        report.table_key,
        report.target_name,
        report.missing_columns,
        report.extra_columns,
        report.coerced_columns,
        report.row_count,
    )


@lru_cache(maxsize=512)
def _contract_alignment_plan(
    table_key: str,
    target_name: str | None,
    extras_policy: ExtrasPolicy | None,
) -> ContractAlignmentPlan:
    contract_schema = _arrow_schema_for_table(table_key, extras_policy=extras_policy)
    return ContractAlignmentPlan(
        table_key=table_key,
        target_name=target_name,
        contract_schema=contract_schema,
        extras_policy=extras_policy,
    )


def _resolve_alignment_policy(
    *,
    table_key: str,
    target_name: str | None,
    policy: ContractPolicy | None,
    extras_policy: ExtrasPolicy | None,
) -> ContractPolicy:
    resolved_policy = policy
    if resolved_policy is None and target_name is not None:
        contract = require_contract_for_target(
            table_key=table_key,
            target_name=target_name,
        )
        resolved_policy = contract.policy
    if resolved_policy is None:
        resolved_policy = ContractPolicy()
    if extras_policy is None:
        return resolved_policy
    if resolved_policy.extras_policy in {None, extras_policy}:
        return replace(resolved_policy, extras_policy=extras_policy)
    msg = (
        "extras_policy conflicts with ContractPolicy extras_policy: "
        f"{extras_policy!r} vs {resolved_policy.extras_policy!r}"
    )
    raise ValueError(msg)


def _assert_schema_types_match(
    contract_schema: pa.Schema,
    incoming_schema: pa.Schema,
) -> None:
    mismatched: list[tuple[str, pa.DataType, pa.DataType]] = []
    incoming_names = set(incoming_schema.names)
    for field in contract_schema:
        if field.name not in incoming_names:
            continue
        incoming_field = incoming_schema.field(field.name)
        if incoming_field.type != field.type:
            mismatched.append((field.name, incoming_field.type, field.type))
    if not mismatched:
        return
    details = ", ".join(
        f"{name} ({incoming} -> {expected})" for name, incoming, expected in mismatched
    )
    msg = f"Contract type coercion disabled; mismatched columns: {details}"
    raise ValueError(msg)


def _arrow_schema_for_table(
    table_key: str,
    *,
    extras_policy: ExtrasPolicy | None = None,
) -> pa.Schema:
    schema_service = get_schema_service()
    if extras_policy is None:
        arrow_schema = schema_service.get_arrow_schema(table_key)
        if arrow_schema is not None:
            return arrow_schema
    table_schema = schema_service.require_table_schema(table_key)
    metadata = None if extras_policy is None else ArrowSchemaMetadata(extras_policy=extras_policy)
    return arrow_contract_for_table_schema(table_schema=table_schema, metadata=metadata)


def align_reader_to_contract(
    table_key: str,
    reader: pa.RecordBatchReader,
    *,
    options: AlignmentOptions | None = None,
    **overrides: Unpack[AlignmentOverrides],
) -> pa.RecordBatchReader:
    """Align an Arrow reader to the contract schema for a table.

    Returns
    -------
    pa.RecordBatchReader
        Reader aligned to the contract schema.
    """
    resolved_options = _merge_alignment_options(options, overrides)
    resolved_policy = _resolve_alignment_policy(
        table_key=table_key,
        target_name=resolved_options.target_name,
        policy=resolved_options.policy,
        extras_policy=resolved_options.extras_policy,
    )
    plan = _contract_alignment_plan(
        table_key=table_key,
        target_name=resolved_options.target_name,
        extras_policy=resolved_policy.extras_policy,
    )
    if resolved_options.reporter is not None:
        report = _build_alignment_report(
            table_key=table_key,
            target_name=resolved_options.target_name,
            contract_schema=plan.contract_schema,
            incoming_schema=reader.schema,
            row_count=None,
        )
        resolved_options.reporter(report)
    if not resolved_policy.coerce_types:
        _assert_schema_types_match(plan.contract_schema, reader.schema)
    return _align_reader(reader, plan.contract_schema, extras_policy=plan.extras_policy)


def align_table_to_contract(
    table_key: str,
    table: pa.Table,
    *,
    options: AlignmentOptions | None = None,
    **overrides: Unpack[AlignmentOverrides],
) -> pa.Table:
    """Align an Arrow table to the contract schema for a table.

    Returns
    -------
    pa.Table
        Arrow table aligned to the contract schema.
    """
    resolved_options = _merge_alignment_options(options, overrides)
    resolved_policy = _resolve_alignment_policy(
        table_key=table_key,
        target_name=resolved_options.target_name,
        policy=resolved_options.policy,
        extras_policy=resolved_options.extras_policy,
    )
    plan = _contract_alignment_plan(
        table_key=table_key,
        target_name=resolved_options.target_name,
        extras_policy=resolved_policy.extras_policy,
    )
    if resolved_options.reporter is not None:
        report = _build_alignment_report(
            table_key=table_key,
            target_name=resolved_options.target_name,
            contract_schema=plan.contract_schema,
            incoming_schema=table.schema,
            row_count=table.num_rows,
        )
        resolved_options.reporter(report)
    if table.schema.equals(plan.contract_schema, check_metadata=True):
        return table
    if table.schema.equals(plan.contract_schema, check_metadata=False):
        return table.replace_schema_metadata(plan.contract_schema.metadata)
    if not resolved_policy.coerce_types:
        _assert_schema_types_match(plan.contract_schema, table.schema)
    reader = table_to_reader(table, batch_size=None)
    aligned = _align_reader(reader, plan.contract_schema, extras_policy=plan.extras_policy)
    return reader_to_table(aligned)


def align_tabular_to_contract(
    table_key: str,
    value: InferableTabularInput,
    *,
    options: AlignmentOptions | None = None,
    **overrides: Unpack[AlignmentOverrides],
) -> InferableTabularInput:
    """Align an inferable tabular input to the contract schema.

    Returns
    -------
    InferableTabularInput
        Aligned tabular input, preserving the input type when possible.
    """
    if isinstance(value, pa.RecordBatchReader):
        return align_reader_to_contract(
            table_key,
            value,
            options=options,
            **overrides,
        )
    if isinstance(value, pa.Table):
        return align_table_to_contract(
            table_key,
            value,
            options=options,
            **overrides,
        )
    if isinstance(value, pl.DataFrame):
        aligned = align_table_to_contract(
            table_key,
            value.to_arrow(),
            options=options,
            **overrides,
        )
        return table_to_frame(aligned)
    if isinstance(value, pl.LazyFrame):
        reader = lazyframe_to_reader(value)
        aligned = align_reader_to_contract(
            table_key,
            reader,
            options=options,
            **overrides,
        )
        return arrow_reader_to_lazyframe(aligned)
    reader = tabular_to_arrow_reader(value)
    return align_reader_to_contract(
        table_key,
        reader,
        options=options,
        **overrides,
    )


def align_reader_to_contract_context(
    context: ContractedTableContext,
    reader: pa.RecordBatchReader,
    *,
    reporter: AlignmentReporter | None = None,
) -> pa.RecordBatchReader:
    """Align a reader using a pre-resolved contract context.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader aligned to the contract schema.
    """
    options = _alignment_options_from_context(context, reporter=reporter)
    return align_reader_to_contract(context.contract.table_key, reader, options=options)


def align_table_to_contract_context(
    context: ContractedTableContext,
    table: pa.Table,
    *,
    reporter: AlignmentReporter | None = None,
) -> pa.Table:
    """Align a table using a pre-resolved contract context.

    Returns
    -------
    pyarrow.Table
        Table aligned to the contract schema.
    """
    options = _alignment_options_from_context(context, reporter=reporter)
    return align_table_to_contract(context.contract.table_key, table, options=options)


def align_tabular_to_contract_context(
    context: ContractedTableContext,
    value: InferableTabularInput,
    *,
    reporter: AlignmentReporter | None = None,
) -> InferableTabularInput:
    """Align a tabular input using a pre-resolved contract context.

    Returns
    -------
    InferableTabularInput
        Aligned tabular input, preserving the input type when possible.
    """
    options = _alignment_options_from_context(context, reporter=reporter)
    return align_tabular_to_contract(context.contract.table_key, value, options=options)


def concat_tables_unified(tables: Sequence[pa.Table]) -> pa.Table:
    """Concatenate tables after unifying schemas.

    Returns
    -------
    pyarrow.Table
        Concatenated Arrow table with a unified schema.
    """
    return _concat_tables_unified(tables)


def arrow_join_frames(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    spec: ArrowJoinSpec | JoinSpec,
    options: ArrowJoinOptions | None = None,
) -> pl.DataFrame:
    """Collect, join in Arrow, and return a Polars DataFrame.

    Returns
    -------
    pl.DataFrame
        Joined Polars DataFrame.
    """
    resolved_spec = spec if isinstance(spec, ArrowJoinSpec) else _arrow_spec_from_join_spec(spec)
    left_table = _arrow_table_from_frame(left)
    right_table = _arrow_table_from_frame(right)
    try:
        joined = arrow_join_tables(
            left_table,
            right_table,
            spec=resolved_spec,
            options=options,
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        return _polars_join_fallback(left, right, spec=resolved_spec)
    return table_to_frame(joined)


def arrow_join_lazyframes(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    *,
    spec: JoinSpec | ArrowJoinSpec | None = None,
    options: ArrowJoinOptions | None = None,
) -> pl.LazyFrame:
    """Join two LazyFrames via Arrow, returning a LazyFrame.

    Returns
    -------
    pl.LazyFrame
        LazyFrame wrapping the Arrow join result.
    """
    if spec is None:
        resolved = _arrow_spec_from_join_spec(JoinSpec())
    elif isinstance(spec, ArrowJoinSpec):
        resolved = spec
    else:
        resolved = _arrow_spec_from_join_spec(spec)
    joined = arrow_join_frames(
        left,
        right,
        spec=resolved,
        options=options,
    )
    return joined.lazy()


iter_array_values = _iter_array_values
iter_rows = _iter_rows


def group_list_or_polars(
    table: pa.Table,
    *,
    keys: Sequence[str],
    value_col: str,
    maintain_order: bool = False,
) -> pa.Table:
    """Group rows into list aggregates, falling back to Polars when needed.

    Returns
    -------
    pa.Table
        Grouped table with list aggregates.
    """
    try:
        return table.group_by(list(keys)).aggregate([(value_col, "list")])
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        frame = cast("pl.DataFrame", pl.from_arrow(table))
        aggregated = (
            frame.lazy()
            .group_by(list(keys), maintain_order=maintain_order)
            .agg(pl.col(value_col).implode().alias(value_col))
            .collect()
        )
        return aggregated.to_arrow()


def write_json_streaming(reader: pa.RecordBatchReader, output_path: str | Path) -> None:
    """Write JSON using streaming Arrow record batches."""
    writer = require_json_writer()
    writer(reader, str(output_path))


def json_writer_available() -> bool:
    """Return whether pyarrow JSON streaming support is available.

    Returns
    -------
    bool
        True when the JSON writer exists in pyarrow.json.
    """
    return getattr(paj, "write_json", None) is not None


def require_json_writer() -> Callable[[pa.RecordBatchReader, str], None]:
    """Return the pyarrow JSON writer or raise when unavailable.

    Returns
    -------
    Callable[[pa.RecordBatchReader, str], None]
        pyarrow JSON writer function.

    Raises
    ------
    AttributeError
        If the pyarrow JSON writer is unavailable.
    """
    writer = getattr(paj, "write_json", None)
    if writer is None:
        msg = "pyarrow.json.write_json is unavailable"
        raise AttributeError(msg)
    return cast("Callable[[pa.RecordBatchReader, str], None]", writer)


def write_json_streaming_table(
    table: pa.Table,
    output_path: str | Path,
    *,
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE,
) -> None:
    """Write JSON from an Arrow table using streaming record batches."""
    reader = table_to_reader(table, batch_size=batch_size)
    write_json_streaming(reader, output_path)


__all__ = [
    "AlignmentOptions",
    "AlignmentOverrides",
    "AlignmentReport",
    "AlignmentReporter",
    "ArrowJoinOptions",
    "ArrowJoinSpec",
    "HashJoinSpec",
    "JoinFilterClause",
    "ParquetScanOptions",
    "Plan",
    "align_reader_to_contract",
    "align_reader_to_contract_context",
    "align_table_to_contract",
    "align_table_to_contract_context",
    "align_tabular_to_contract",
    "align_tabular_to_contract_context",
    "arrow_join_frames",
    "arrow_join_lazyframes",
    "arrow_join_tables",
    "arrow_table_from_lazyframe",
    "arrow_table_from_tabular",
    "build_join_options",
    "combine_join_filters",
    "concat_tables_unified",
    "dedupe_table_for_table",
    "dedupe_tabular",
    "drain_alignment_diagnostics",
    "emit_alignment_report",
    "ensure_array",
    "group_list_or_polars",
    "index_in",
    "join_filter_expr",
    "json_writer_available",
    "normalize_binary_view_array",
    "normalize_string_view_array",
    "normalize_table_for_compute",
    "normalize_table_for_join",
    "pop_alignment_report",
    "record_alignment_diagnostic",
    "record_alignment_report",
    "require_json_writer",
    "resolve_join_filter_field",
    "scan_parquet_dataset",
    "scan_parquet_table",
    "table_to_reader",
    "take_by_key",
    "value_set_array",
    "write_json_streaming",
    "write_json_streaming_table",
]
