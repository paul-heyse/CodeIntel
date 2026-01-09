"""List explode helpers for Arrow tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.compute_helpers import call_compute, require_array

NullListPolicy = Literal["error", "empty"]
NullChildPolicy = Literal["drop", "error"]


@dataclass(frozen=True, slots=True)
class ExplodeSpec:
    """Configuration for exploding list payloads."""

    src_col: str
    dst_list_col: str
    repeat_cols: Sequence[str] = ()
    aligned_list_cols: Sequence[str] = ()
    null_list_policy: NullListPolicy = "error"
    null_child_policy: NullChildPolicy = "drop"
    enforce_parent_valid: bool = True
    error_context_cols: Sequence[str] = ()


@dataclass(frozen=True, slots=True)
class ExplodeResult:
    """Result of exploding a list column."""

    good: pa.Table
    errors: pa.Table


@dataclass(frozen=True, slots=True)
class ErrorSpec:
    """Error metadata for explode error tables."""

    error_code: str
    column: str
    detail: str


@dataclass(frozen=True, slots=True)
class ErrorContext:
    """Context for building explode error tables."""

    table: pa.Table
    row_id: pa.Array | pa.ChunkedArray
    error_context_cols: Sequence[str]


@dataclass(frozen=True, slots=True)
class ParentFilterResult:
    """Parent filtering output for list explode operations."""

    table: pa.Table
    row_id: pa.Array | pa.ChunkedArray
    errors: Sequence[pa.Table]


def explode_edges(
    table: pa.Table,
    *,
    spec: ExplodeSpec,
) -> ExplodeResult:
    """Explode list columns into edge rows.

    Parameters
    ----------
    table
        Source table with list columns.
    spec
        Explode specification.

    Returns
    -------
    ExplodeResult
        Exploded rows plus error rows.
    """
    if table.num_rows == 0:
        return _empty_explode_result(table, spec)

    row_id = pa.arange(0, table.num_rows)
    context = ErrorContext(
        table=table,
        row_id=row_id,
        error_context_cols=spec.error_context_cols,
    )
    parents = _filter_parents(context, spec)
    if parents.table.num_rows == 0:
        empty_good = _empty_explode_good(table, spec)
        errors = _concat_errors(parents.errors, table, spec.error_context_cols)
        return ExplodeResult(good=empty_good, errors=errors)

    exploded, child_errors = _explode_children(parents, spec)
    errors = _concat_errors([*parents.errors, *child_errors], table, spec.error_context_cols)
    return ExplodeResult(good=exploded, errors=errors)


def explode_edges_with_aligned_lists(
    table: pa.Table,
    *,
    spec: ExplodeSpec,
) -> ExplodeResult:
    """Explode a list column while enforcing aligned list payloads.

    Parameters
    ----------
    table
        Table containing the list payload column.
    spec
        Explode specification including alignment and null policies.

    Returns
    -------
    ExplodeResult
        Exploded rows plus error rows.
    """
    return explode_edges(table, spec=spec)


def explode_list_struct(
    table: pa.Table,
    *,
    list_col: str,
    parent_cols: Sequence[str],
    struct_fields: Mapping[str, str],
) -> pa.Table:
    """Explode a list<struct> column into a row-per-element table.

    Parameters
    ----------
    table
        Input table with a list<struct> column.
    list_col
        Name of the list<struct> column to explode.
    parent_cols
        Columns to repeat for each exploded element.
    struct_fields
        Mapping of struct field name to output column name.

    Returns
    -------
    pyarrow.Table
        Table with repeated parent columns and extracted struct fields.
    """
    lists = table[list_col]
    parent_idx = _list_parent_indices(lists)
    flat_struct = _list_flatten(lists)

    cols: dict[str, pa.Array | pa.ChunkedArray] = {}
    for name in parent_cols:
        cols[name] = pc.take(table[name], parent_idx)

    for field_name, out_name in struct_fields.items():
        cols[out_name] = _struct_field(flat_struct, field_name)

    return pa.table(cols)


def _empty_explode_result(table: pa.Table, spec: ExplodeSpec) -> ExplodeResult:
    empty_good = _empty_explode_good(table, spec)
    empty_errors = _empty_error_table(table, spec.error_context_cols)
    return ExplodeResult(good=empty_good, errors=empty_errors)


def _empty_explode_good(table: pa.Table, spec: ExplodeSpec) -> pa.Table:
    schema = _explode_schema(table, spec)
    return pa.Table.from_batches([], schema=schema)


def _filter_parents(context: ErrorContext, spec: ExplodeSpec) -> ParentFilterResult:
    dst_lists = context.table[spec.dst_list_col]
    errors: list[pa.Table] = []

    null_mask = None
    if spec.null_list_policy == "error":
        null_mask = _fill_null_false(_is_null(dst_lists))
        errors.append(
            _error_table_for_parent_rows(
                context,
                mask=null_mask,
                spec=ErrorSpec(
                    error_code="NULL_LIST",
                    column=spec.dst_list_col,
                    detail="null parent list",
                ),
            )
        )

    alignment_mask = _list_alignment_mask(context.table, spec)
    if alignment_mask is not None:
        errors.append(
            _error_table_for_parent_rows(
                context,
                mask=alignment_mask,
                spec=ErrorSpec(
                    error_code="MISALIGNED_LIST_COLUMNS",
                    column=spec.dst_list_col,
                    detail="aligned list lengths differ",
                ),
            )
        )

    bad_parent_mask = _combine_masks([null_mask, alignment_mask])
    if bad_parent_mask is None:
        return ParentFilterResult(table=context.table, row_id=context.row_id, errors=errors)

    good_parent_mask = _invert(bad_parent_mask)
    filtered_table = context.table.filter(good_parent_mask)
    filtered_row_id = _filter_array(context.row_id, good_parent_mask)
    return ParentFilterResult(
        table=filtered_table,
        row_id=filtered_row_id,
        errors=errors,
    )


def _explode_children(
    parents: ParentFilterResult,
    spec: ExplodeSpec,
) -> tuple[pa.Table, Sequence[pa.Table]]:
    lists = parents.table[spec.dst_list_col]
    parent_idx = _list_parent_indices(lists)
    dst_flat = _list_flatten(lists)

    cols = _build_exploded_columns(parents.table, parent_idx, spec)
    parent_valid_rep = None
    if spec.enforce_parent_valid:
        parent_valid_rep = _parent_valid_mask(lists, parent_idx)

    if spec.null_child_policy == "drop":
        child_valid = _fill_null_false(_is_valid(dst_flat))
        if parent_valid_rep is not None:
            child_valid = _and(child_valid, parent_valid_rep)
        exploded = _filtered_exploded_table(cols, spec.dst_list_col, dst_flat, child_valid)
        return exploded, []

    exploded = _unfiltered_exploded_table(cols, spec.dst_list_col, dst_flat)
    if parent_valid_rep is not None:
        exploded = exploded.filter(parent_valid_rep)

    error_context = ErrorContext(
        table=parents.table,
        row_id=parents.row_id,
        error_context_cols=spec.error_context_cols,
    )
    child_errors = _child_errors(
        error_context,
        parent_idx=parent_idx,
        dst_flat=dst_flat,
        column=spec.dst_list_col,
    )
    return exploded, child_errors


def _build_exploded_columns(
    table: pa.Table,
    parent_idx: pa.Array | pa.ChunkedArray,
    spec: ExplodeSpec,
) -> dict[str, pa.Array | pa.ChunkedArray]:
    cols: dict[str, pa.Array | pa.ChunkedArray] = {
        spec.src_col: pc.take(table[spec.src_col], parent_idx),
    }
    for name in spec.repeat_cols:
        cols[name] = pc.take(table[name], parent_idx)
    for name in spec.aligned_list_cols:
        cols[name] = _list_flatten(table[name])
    return cols


def _parent_valid_mask(
    lists: pa.Array | pa.ChunkedArray,
    parent_idx: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    parent_valid = _fill_null_false(_is_valid(lists))
    parent_valid_rep = pc.take(parent_valid, parent_idx)
    return _fill_null_false(parent_valid_rep)


def _filtered_exploded_table(
    cols: Mapping[str, pa.Array | pa.ChunkedArray],
    dst_col: str,
    dst_values: pa.Array | pa.ChunkedArray,
    mask: pa.Array | pa.ChunkedArray,
) -> pa.Table:
    filtered_cols = {name: _filter_array(value, mask) for name, value in cols.items()}
    filtered_dst = _filter_array(dst_values, mask)
    return _unfiltered_exploded_table(filtered_cols, dst_col, filtered_dst)


def _unfiltered_exploded_table(
    cols: Mapping[str, pa.Array | pa.ChunkedArray],
    dst_col: str,
    dst_values: pa.Array | pa.ChunkedArray,
) -> pa.Table:
    exploded = pa.table(dict(cols))
    return exploded.append_column(dst_col, dst_values)


def _child_errors(
    context: ErrorContext,
    *,
    parent_idx: pa.Array | pa.ChunkedArray,
    dst_flat: pa.Array | pa.ChunkedArray,
    column: str,
) -> list[pa.Table]:
    child_null_mask = _fill_null_false(_is_null(dst_flat))
    return [
        _error_table_for_child_rows(
            context,
            parent_idx=parent_idx,
            mask=child_null_mask,
            spec=ErrorSpec(
                error_code="NULL_CHILD_VALUE",
                column=column,
                detail="null child value",
            ),
        )
    ]


def _explode_schema(table: pa.Table, spec: ExplodeSpec) -> pa.Schema:
    fields: list[pa.Field] = []
    base = table.schema
    for name in (spec.src_col, *spec.repeat_cols):
        if name in base.names:
            fields.append(base.field(name))
        else:
            fields.append(pa.field(name, pa.null()))
    for name in spec.aligned_list_cols:
        if name in base.names:
            field = base.field(name)
            fields.append(pa.field(name, _list_value_type(field.type)))
        else:
            fields.append(pa.field(name, pa.null()))
    if spec.dst_list_col in base.names:
        dst_field = base.field(spec.dst_list_col)
        fields.append(pa.field(spec.dst_list_col, _list_value_type(dst_field.type)))
    else:
        fields.append(pa.field(spec.dst_list_col, pa.null()))
    return pa.schema(fields)


def _list_alignment_mask(
    table: pa.Table,
    spec: ExplodeSpec,
) -> pa.Array | pa.ChunkedArray | None:
    if not spec.aligned_list_cols:
        return None
    dst_len = _list_value_length(table[spec.dst_list_col])
    bad_mask: pa.Array | pa.ChunkedArray | None = None
    for name in spec.aligned_list_cols:
        aligned_len = _list_value_length(table[name])
        equal_mask = _equal(dst_len, aligned_len)
        equal_mask = _fill_null_false(equal_mask)
        mismatch = _invert(equal_mask)
        bad_mask = mismatch if bad_mask is None else _or(bad_mask, mismatch)
    return bad_mask


def _combine_masks(
    masks: Sequence[pa.Array | pa.ChunkedArray | None],
) -> pa.Array | pa.ChunkedArray | None:
    combined: pa.Array | pa.ChunkedArray | None = None
    for mask in masks:
        if mask is None:
            continue
        combined = mask if combined is None else _or(combined, mask)
    return combined


def _fill_null_false(mask: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _fill_null(mask, fill_value=False)


def _filter_array(
    values: pa.Array | pa.ChunkedArray,
    mask: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    return require_array(call_compute("filter", [values, mask]), name="filter")


def _error_table_for_parent_rows(
    context: ErrorContext,
    *,
    mask: pa.Array | pa.ChunkedArray | None,
    spec: ErrorSpec,
) -> pa.Table:
    if mask is None:
        return _empty_error_table(context.table, context.error_context_cols)
    indices = _indices_nonzero(mask)
    if len(indices) == 0:
        return _empty_error_table(context.table, context.error_context_cols)
    return _build_parent_error_table(context, indices=indices, spec=spec)


def _error_table_for_child_rows(
    context: ErrorContext,
    *,
    parent_idx: pa.Array | pa.ChunkedArray,
    mask: pa.Array | pa.ChunkedArray,
    spec: ErrorSpec,
) -> pa.Table:
    indices = _indices_nonzero(mask)
    if len(indices) == 0:
        return _empty_error_table(context.table, context.error_context_cols)
    return _build_child_error_table(
        context,
        parent_idx=parent_idx,
        indices=indices,
        spec=spec,
    )


def _build_parent_error_table(
    context: ErrorContext,
    *,
    indices: pa.Array | pa.ChunkedArray,
    spec: ErrorSpec,
) -> pa.Table:
    return _build_error_table(
        context,
        row_id=context.row_id,
        indices=indices,
        spec=spec,
    )


def _build_child_error_table(
    context: ErrorContext,
    *,
    parent_idx: pa.Array | pa.ChunkedArray,
    indices: pa.Array | pa.ChunkedArray,
    spec: ErrorSpec,
) -> pa.Table:
    parent_rows = pc.take(context.row_id, parent_idx)
    columns = _error_columns(
        row_id=parent_rows,
        indices=indices,
        spec=spec,
    )
    for name in context.error_context_cols:
        if name not in context.table.column_names:
            continue
        parent_values = pc.take(context.table[name], parent_idx)
        columns[name] = pc.take(parent_values, indices)
    return pa.table(columns)


def _build_error_table(
    context: ErrorContext,
    *,
    row_id: pa.Array | pa.ChunkedArray,
    indices: pa.Array | pa.ChunkedArray,
    spec: ErrorSpec,
) -> pa.Table:
    columns = _error_columns(row_id=row_id, indices=indices, spec=spec)
    for name in context.error_context_cols:
        if name in context.table.column_names:
            columns[name] = pc.take(context.table[name], indices)
    return pa.table(columns)


def _error_columns(
    *,
    row_id: pa.Array | pa.ChunkedArray,
    indices: pa.Array | pa.ChunkedArray,
    spec: ErrorSpec,
) -> dict[str, pa.Array | pa.ChunkedArray]:
    count = len(indices)
    return {
        "row_id": pc.take(row_id, indices),
        "error_code": pa.array([spec.error_code] * count, type=pa.string()),
        "column": pa.array([spec.column] * count, type=pa.string()),
        "detail": pa.array([spec.detail] * count, type=pa.string()),
    }


def _empty_error_table(table: pa.Table, error_context_cols: Sequence[str]) -> pa.Table:
    fields = [
        pa.field("row_id", pa.int64()),
        pa.field("error_code", pa.string()),
        pa.field("column", pa.string()),
        pa.field("detail", pa.string()),
    ]
    fields.extend(
        [table.schema.field(name) for name in error_context_cols if name in table.column_names]
    )
    return pa.Table.from_batches([], schema=pa.schema(fields))


def _concat_errors(
    errors: Sequence[pa.Table],
    table: pa.Table,
    error_context_cols: Sequence[str],
) -> pa.Table:
    non_empty = [error for error in errors if error.num_rows > 0]
    if not non_empty:
        return _empty_error_table(table, error_context_cols)
    return pa.concat_tables(non_empty, promote_options="default")


def _compute_array(name: str, args: Sequence[object]) -> pa.Array | pa.ChunkedArray:
    return require_array(call_compute(name, list(args)), name=name)


def _is_null(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _compute_array("is_null", [values])


def _is_valid(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _compute_array("is_valid", [values])


def _list_parent_indices(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _compute_array("list_parent_indices", [values])


def _list_flatten(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _compute_array("list_flatten", [values])


def _list_value_length(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _compute_array("list_value_length", [values])


def _indices_nonzero(mask: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _compute_array("indices_nonzero", [mask])


def _struct_field(
    values: pa.Array | pa.ChunkedArray,
    field_name: str,
) -> pa.Array | pa.ChunkedArray:
    return _compute_array("struct_field", [values, field_name])


def _equal(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    return _compute_array("equal", [left, right])


def _invert(mask: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _compute_array("invert", [mask])


def _or(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    return _compute_array("or", [left, right])


def _and(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    return _compute_array("and", [left, right])


def _fill_null(
    mask: pa.Array | pa.ChunkedArray,
    *,
    fill_value: bool,
) -> pa.Array | pa.ChunkedArray:
    return _compute_array("fill_null", [mask, fill_value])


def _is_list_type(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    )


def _is_list_view_type(data_type: pa.DataType) -> bool:
    is_list_view = getattr(pa.types, "is_list_view", None)
    is_large_list_view = getattr(pa.types, "is_large_list_view", None)
    return bool(
        (callable(is_list_view) and is_list_view(data_type))
        or (callable(is_large_list_view) and is_large_list_view(data_type))
    )


def _list_value_type(data_type: pa.DataType) -> pa.DataType:
    if _is_list_type(data_type) or _is_list_view_type(data_type):
        return data_type.value_type
    return data_type


__all__ = [
    "ExplodeResult",
    "ExplodeSpec",
    "explode_edges",
    "explode_edges_with_aligned_lists",
    "explode_list_struct",
]
