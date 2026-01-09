"""Schema-driven row ordering helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.core.columnar.rows import ColumnarRowBuffer, columnar_buffer_for_table_key
from codeintel.core.schemas.row_models import columns_for_table_key, row_serializer_for_table_schema
from codeintel.core.schemas.table_registry import TABLE_SCHEMAS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


def row_tuple_for_table(
    table_key: str,
    row: Mapping[str, object],
) -> tuple[object, ...]:
    """Build a row tuple ordered by the table schema columns.

    Returns
    -------
    tuple[object, ...]
        Tuple ordered according to the table schema.
    """
    columns = _columns_for_table(table_key)
    return _row_tuple(columns, row, table_key=table_key)


def rows_to_tuples_for_table(
    table_key: str,
    rows: Iterable[Mapping[str, object]],
) -> list[tuple[object, ...]]:
    """Build ordered row tuples for a sequence of row mappings.

    Returns
    -------
    list[tuple[object, ...]]
        Ordered row tuples aligned to the schema.
    """
    columns = _columns_for_table(table_key)
    return [_row_tuple(columns, row, table_key=table_key) for row in rows]


def buffer_for_table(table_key: str) -> ColumnarRowBuffer:
    """Return a columnar buffer seeded from the table schema.

    Returns
    -------
    ColumnarRowBuffer
        Buffer ready to accept row mappings.
    """
    return columnar_buffer_for_table_key(table_key)


def buffer_from_rows(
    table_key: str,
    rows: Iterable[Mapping[str, object]],
) -> ColumnarRowBuffer:
    """Build a columnar buffer from row mappings.

    Returns
    -------
    ColumnarRowBuffer
        Buffer containing appended rows.
    """
    buffer = columnar_buffer_for_table_key(table_key)
    for row in rows:
        buffer.append(row)
    return buffer


def _columns_for_table(table_key: str) -> tuple[str, ...]:
    columns = columns_for_table_key(table_key)
    if not columns:
        msg = f"No schema columns registered for {table_key}"
        raise ValueError(msg)
    return tuple(columns)


def _row_tuple(
    columns: Sequence[str],
    row: Mapping[str, object],
    *,
    table_key: str,
) -> tuple[object, ...]:
    missing = [column for column in columns if column not in row]
    if missing:
        msg = f"Row missing columns for {table_key}: {missing}"
        raise KeyError(msg)
    table_schema = TABLE_SCHEMAS.get(table_key)
    if table_schema is None:
        msg = f"No TableSchema registered for {table_key}"
        raise KeyError(msg)
    serializer = row_serializer_for_table_schema(table_schema=table_schema)
    return serializer(row)
