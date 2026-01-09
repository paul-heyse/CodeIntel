"""Arrow compute kernels for graph assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.validation.schema_constraints import is_list_like


@dataclass(frozen=True, slots=True)
class ExplodeEdgesResult:
    """Result of exploding list-based edges."""

    edges: pa.Table
    invalid_parents: pa.Table


def explode_edges(
    table: pa.Table,
    *,
    src_col: str,
    dst_list_col: str,
    repeat_cols: Sequence[str] = (),
    src_name: str = "src_id",
    dst_name: str = "dst_id",
) -> pa.Table:
    """Explode list-valued destination columns into edges."""
    _ensure_list_column(table, dst_list_col)
    parent_idx = pc.list_parent_indices(table[dst_list_col])
    dst_flat = pc.list_flatten(table[dst_list_col])
    output: dict[str, pa.Array | pa.ChunkedArray] = {
        src_name: pc.take(table[src_col], parent_idx),
        dst_name: dst_flat,
    }
    _add_repeated_columns(output, table, parent_idx, repeat_cols)
    return pa.table(output)


def explode_edges_with_aligned_lists(
    table: pa.Table,
    *,
    src_col: str,
    dst_list_col: str,
    aligned_list_cols: Sequence[str],
    repeat_cols: Sequence[str] = (),
    src_name: str = "src_id",
    dst_name: str = "dst_id",
    nulls_match: bool = True,
) -> ExplodeEdgesResult:
    """Explode list-valued edges with aligned list attributes."""
    _ensure_list_column(table, dst_list_col)
    for column in aligned_list_cols:
        _ensure_list_column(table, column)
    mismatch_mask = _alignment_mismatch_mask(
        table,
        dst_list_col=dst_list_col,
        aligned_list_cols=aligned_list_cols,
        nulls_match=nulls_match,
    )
    if mismatch_mask is None:
        return ExplodeEdgesResult(
            edges=_explode_edges_table(
                table,
                src_col=src_col,
                dst_list_col=dst_list_col,
                aligned_list_cols=aligned_list_cols,
                repeat_cols=repeat_cols,
                src_name=src_name,
                dst_name=dst_name,
            ),
            invalid_parents=_empty_slice(table),
        )
    invalid_parents = table.filter(mismatch_mask)
    good_table = table.filter(pc.invert(mismatch_mask))
    edges = _explode_edges_table(
        good_table,
        src_col=src_col,
        dst_list_col=dst_list_col,
        aligned_list_cols=aligned_list_cols,
        repeat_cols=repeat_cols,
        src_name=src_name,
        dst_name=dst_name,
    )
    return ExplodeEdgesResult(edges=edges, invalid_parents=invalid_parents)


def _explode_edges_table(
    table: pa.Table,
    *,
    src_col: str,
    dst_list_col: str,
    aligned_list_cols: Sequence[str],
    repeat_cols: Sequence[str],
    src_name: str,
    dst_name: str,
) -> pa.Table:
    parent_idx = pc.list_parent_indices(table[dst_list_col])
    output: dict[str, pa.Array | pa.ChunkedArray] = {
        src_name: pc.take(table[src_col], parent_idx),
        dst_name: pc.list_flatten(table[dst_list_col]),
    }
    _add_repeated_columns(output, table, parent_idx, repeat_cols)
    for column in aligned_list_cols:
        if column in output:
            continue
        output[column] = pc.list_flatten(table[column])
    return pa.table(output)


def _alignment_mismatch_mask(
    table: pa.Table,
    *,
    dst_list_col: str,
    aligned_list_cols: Sequence[str],
    nulls_match: bool,
) -> pa.Array | pa.ChunkedArray | None:
    if not aligned_list_cols:
        return None
    dst_len = pc.list_value_length(table[dst_list_col])
    mismatch: pa.Array | pa.ChunkedArray | None = None
    for column in aligned_list_cols:
        col_len = pc.list_value_length(table[column])
        eq = pc.equal(dst_len, col_len)
        eq = pc.fill_null(eq, nulls_match)
        col_mismatch = pc.invert(eq)
        mismatch = col_mismatch if mismatch is None else pc.or_(mismatch, col_mismatch)
    return mismatch


def _add_repeated_columns(
    output: dict[str, pa.Array | pa.ChunkedArray],
    table: pa.Table,
    parent_idx: pa.Array | pa.ChunkedArray,
    repeat_cols: Sequence[str],
) -> None:
    for column in repeat_cols:
        if column in output:
            continue
        output[column] = pc.take(table[column], parent_idx)


def _ensure_list_column(table: pa.Table, column: str) -> None:
    if column not in table.column_names:
        msg = f"Expected list column {column!r} in table"
        raise ValueError(msg)
    if not is_list_like(table[column].type):
        msg = f"Expected list-like type for column {column!r}"
        raise TypeError(msg)


def _empty_slice(table: pa.Table) -> pa.Table:
    return table.slice(0, 0)


__all__ = ["ExplodeEdgesResult", "explode_edges", "explode_edges_with_aligned_lists"]
