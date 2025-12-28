"""Columnar test helpers for building Arrow tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeGuard, cast

import pyarrow as pa

from codeintel.core.columnar.rows import (
    ColumnarRowBuffer,
    ColumnarRows,
    columnar_buffer_for_table_key,
    columnar_row_count,
)
from codeintel.storage.warehouse import Warehouse

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.storage.warehouse import MaterializationResult, MaterializeOptions


def arrow_table_for_rows(
    table_key: str,
    rows: Sequence[tuple[object, ...]] | ColumnarRows,
    *,
    columns: Sequence[str] | None = None,
) -> pa.Table:
    """Create an Arrow table from row tuples or columnar rows.

    Parameters
    ----------
    table_key
        Table key used to resolve schema order.
    rows
        Row tuples or columnar rows.
    columns
        Optional column order for tuple rows. Defaults to schema order.

    Returns
    -------
    pyarrow.Table
        Arrow table with schema-aligned columns.
    """
    buffer = columnar_buffer_for_table_key(table_key)
    resolved_columns = tuple(columns) if columns else buffer.columns
    if not resolved_columns:
        return pa.table({})
    buffer = _slice_buffer(buffer, columns=resolved_columns)
    buffer = _buffer_from_rows(buffer, rows, columns=resolved_columns)
    if buffer.row_count == 0:
        return pa.table({name: [] for name in buffer.columns})
    return pa.table(buffer.data)


def materialize_table_from_rows(
    warehouse: Warehouse,
    table_key: str,
    rows: Sequence[tuple[object, ...]] | ColumnarRows,
    *,
    columns: Sequence[str] | None = None,
    options: MaterializeOptions | None = None,
) -> MaterializationResult:
    """Materialize row data via the columnar table path.

    Parameters
    ----------
    warehouse
        Warehouse to use for materialization.
    table_key
        Destination table key.
    rows
        Row tuples or columnar rows to materialize.
    columns
        Optional column order for tuple rows.
    options
        Optional materialization options.

    Returns
    -------
    MaterializationResult
        Result metadata for the write.
    """
    table = arrow_table_for_rows(table_key, rows, columns=columns)
    return warehouse.materialize_table(table_key, table, options=options)


def _buffer_from_rows(
    buffer: ColumnarRowBuffer,
    rows: Sequence[tuple[object, ...]] | ColumnarRows,
    *,
    columns: Sequence[str],
) -> ColumnarRowBuffer:
    if _is_columnar_rows(rows):
        row_count = columnar_row_count(rows)
        for row_idx in range(row_count):
            buffer.append(
                _columnar_row_at_index(
                    rows,
                    columns=buffer.columns,
                    row_idx=row_idx,
                )
            )
        return buffer

    column_index = {name: idx for idx, name in enumerate(columns)}
    for row in rows:
        row_map = {name: row[column_index[name]] for name in buffer.columns}
        buffer.append(row_map)
    return buffer


def _columnar_row_at_index(
    rows: ColumnarRows,
    *,
    columns: Sequence[str],
    row_idx: int,
) -> Mapping[str, object]:
    row_map: dict[str, object] = {}
    for name in columns:
        values = rows.get(name)
        row_map[name] = values[row_idx] if values is not None else None
    return row_map


def _slice_buffer(
    buffer: ColumnarRowBuffer,
    *,
    columns: Sequence[str],
) -> ColumnarRowBuffer:
    column_set = set(columns)
    type_by_name: dict[str, ColumnType] = {
        name: cast("ColumnType", col_type)
        for name, col_type in zip(buffer.columns, buffer.column_types, strict=True)
    }
    missing = column_set.difference(type_by_name)
    if missing:
        msg = f"Unknown columns for materialization: {sorted(missing)}"
        raise KeyError(msg)
    return ColumnarRowBuffer(
        table_key=buffer.table_key,
        columns=tuple(columns),
        column_types=tuple(type_by_name[name] for name in columns),
        data={name: [] for name in columns},
    )


def _is_columnar_rows(
    rows: Sequence[tuple[object, ...]] | ColumnarRows,
) -> TypeGuard[ColumnarRows]:
    return isinstance(rows, dict)


__all__ = ["arrow_table_for_rows", "materialize_table_from_rows"]
