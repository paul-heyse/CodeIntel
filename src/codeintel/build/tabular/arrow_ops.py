"""Arrow-first join and materialization helpers for build pipelines.

Policy
------
- Graph compute modules use Arrow tables/readers end-to-end and call these helpers.
- Polars fallbacks are reserved for legacy or view/export paths only.
- Join keys and cardinality expectations live in `docs/architecture/arrow_join_policy.md`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.json as paj

from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular import array_ops as _array_ops
from codeintel.build.tabular.compute_helpers import scalar_from_compute
from codeintel.build.tabular.conversion import (
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
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.normalization import (
    normalize_table_for_compute as _normalize_table_for_compute,
)
from codeintel.core.columnar.schema_alignment import align_reader_to_contract as _align_reader
from codeintel.core.columnar.schema_ops import concat_tables_unified as _concat_tables_unified
from codeintel.core.columnar.streaming import DatasetScanOptions
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.datasets.arrow_store import scan_dataset
from codeintel.core.datasets.scanner_ops import build_scanner
from codeintel.core.schemas.arrow_gen import (
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
class ParquetScanOptions:
    columns: Sequence[str] | None = None
    repo: str | None = None
    commit: str | None = None
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE


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
        return pc.cast(column, target_type, safe=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
        return column


def _cast_binary_view_column(
    column: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    target_type = _binary_view_cast_type(column.type)
    if target_type == column.type:
        return column
    try:
        return pc.cast(column, target_type, safe=False)
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


def normalize_table_for_join(table: pa.Table) -> pa.Table:
    """Normalize string/binary view types ahead of Arrow joins.

    Returns
    -------
    pyarrow.Table
        Table with view types normalized for join compatibility.
    """
    return _normalize_table_binary_views(_normalize_table_string_views(table))


def normalize_table_for_compute(table: pa.Table) -> pa.Table:
    """Normalize a table for compute-heavy kernels.

    Returns
    -------
    pa.Table
        Table with normalized view types, unified dictionaries, and combined chunks.
    """
    normalized = normalize_table_for_join(table)
    return _normalize_table_for_compute(normalized)


def _ensure_unique_keys(table: pa.Table, keys: Sequence[str], *, label: str) -> None:
    if not keys:
        return
    missing = [key for key in keys if key not in table.column_names]
    if missing:
        msg = f"Missing join keys on {label}: {', '.join(missing)}"
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
    return reader_to_table(reader)


def arrow_table_from_lazyframe(frame: pl.LazyFrame) -> pa.Table:
    """Collect a LazyFrame into an Arrow Table.

    Returns
    -------
    pa.Table
        Materialized Arrow table.
    """
    return reader_to_table(lazyframe_to_reader(frame))


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


def _arrow_schema_for_table(
    table_key: str,
    *,
    extras_policy: ExtrasPolicy | None,
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
    extras_policy: ExtrasPolicy | None = None,
) -> pa.RecordBatchReader:
    """Align an Arrow reader to the contract schema for a table.

    Returns
    -------
    pa.RecordBatchReader
        Reader aligned to the contract schema.
    """
    contract_schema = _arrow_schema_for_table(table_key, extras_policy=extras_policy)
    return _align_reader(reader, contract_schema, extras_policy=extras_policy)


def align_table_to_contract(
    table_key: str,
    table: pa.Table,
    *,
    extras_policy: ExtrasPolicy | None = None,
) -> pa.Table:
    """Align an Arrow table to the contract schema for a table.

    Returns
    -------
    pa.Table
        Arrow table aligned to the contract schema.
    """
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
    aligned = align_reader_to_contract(table_key, reader, extras_policy=extras_policy)
    return reader_to_table(aligned)


def concat_tables_unified(tables: Sequence[pa.Table]) -> pa.Table:
    """Concatenate tables after unifying schemas.

    Returns
    -------
    pyarrow.Table
        Concatenated Arrow table with a unified schema.
    """
    return _concat_tables_unified(tables)


def scan_parquet_dataset(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    options: ParquetScanOptions | None = None,
) -> pa.RecordBatchReader | None:
    """Return a RecordBatchReader for a parquet dataset snapshot.

    Returns
    -------
    pa.RecordBatchReader | None
        RecordBatchReader when a dataset snapshot is available, otherwise None.
    """
    resolved = options or ParquetScanOptions()
    try:
        dataset = scan_dataset(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
    except FileNotFoundError:
        LOG.warning("Dataset snapshot missing for %s@%s", table_key, snapshot_id)
        return None
    except (OSError, ValueError, pa.ArrowInvalid) as exc:
        LOG.warning("Dataset scan failed for %s@%s: %s", table_key, snapshot_id, exc)
        return None

    names = set(dataset.schema.names)
    expression: ds.Expression | None = None
    if resolved.repo is not None and "repo" in names:
        expression = ds.field("repo") == resolved.repo
    if resolved.commit is not None and "commit" in names:
        commit_expr = ds.field("commit") == resolved.commit
        expression = commit_expr if expression is None else expression & commit_expr

    scan_options = DatasetScanOptions(
        batch_size=resolved.batch_size,
        filter_expression=expression,
        columns=tuple(resolved.columns) if resolved.columns is not None else None,
        unify_schemas=True,
    )
    scanner = build_scanner(dataset, options=scan_options)
    return scanner.to_reader()


def scan_parquet_table(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    options: ParquetScanOptions | None = None,
) -> pa.Table | None:
    """Return a materialized Arrow Table for a parquet dataset snapshot.

    Returns
    -------
    pa.Table | None
        Materialized Arrow table when available, otherwise None.
    """
    reader = scan_parquet_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=options,
    )
    if reader is None:
        return None
    return reader_to_table(reader)


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


def iter_array_values(values: pa.Array | pa.ChunkedArray) -> Iterator[object]:
    """Yield Python values from an Arrow array without materializing a full list.

    Yields
    ------
    object
        Python scalar values.
    """
    if isinstance(values, pa.ChunkedArray):
        for chunk in values.iterchunks():
            for item in chunk:
                yield item.as_py()
        return
    for item in values:
        yield item.as_py()


def iter_rows(
    table_or_batch: pa.Table | pa.RecordBatch,
    columns: Sequence[str] | None = None,
) -> Iterator[dict[str, object]]:
    """Yield row dicts from a table or record batch without building a pylist.

    Yields
    ------
    dict[str, object]
        Row dictionaries.
    """
    if isinstance(table_or_batch, pa.Table):
        column_names = list(columns) if columns is not None else list(table_or_batch.column_names)
        if not column_names:
            return
        selected = table_or_batch.select(column_names)
        for batch in selected.to_batches():
            yield from iter_rows(batch, column_names)
        return
    batch = table_or_batch
    column_names = list(columns) if columns is not None else list(batch.schema.names)
    if not column_names:
        return
    arrays = [batch.column(column_name) for column_name in column_names]
    for row_index in range(batch.num_rows):
        yield {
            column_name: arrays[idx][row_index].as_py()
            for idx, column_name in enumerate(column_names)
        }


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
    "ArrowJoinOptions",
    "ArrowJoinSpec",
    "JoinFilterClause",
    "align_reader_to_contract",
    "align_table_to_contract",
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
    "ensure_array",
    "group_list_or_polars",
    "index_in",
    "join_filter_expr",
    "json_writer_available",
    "normalize_binary_view_array",
    "normalize_string_view_array",
    "normalize_table_for_compute",
    "normalize_table_for_join",
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
