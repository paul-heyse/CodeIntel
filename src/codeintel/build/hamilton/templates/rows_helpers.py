"""Row conversion helpers for row-oriented materialization.

These helpers are intentionally **not** Hamilton nodes.
They are small, deterministic utilities that are safe to call inside compute
functions or tests without affecting Hamilton module discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.core.schemas.row_serialization import row_serializer_for_table_key

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def row_to_tuple(
    row: Mapping[str, object],
    columns: tuple[str, ...] | None = None,
    *,
    table_key: str | None = None,
) -> tuple[object, ...]:
    """Convert a mapping row to a tuple in column order.

    Parameters
    ----------
    row
        Row mapping.
    columns
        Column ordering to extract.
    table_key
        Table key used to resolve schema-based column order.

    Returns
    -------
    tuple[object, ...]
        Tuple of row values in column order. Missing keys yield None.

    Raises
    ------
    ValueError
        When neither table_key nor columns are provided.
    """
    if table_key is not None:
        return row_serializer_for_table_key(table_key)(row)
    if columns is None:
        msg = "row_to_tuple requires table_key or columns"
        raise ValueError(msg)
    return tuple(row.get(col) for col in columns)


def rows_to_tuples(
    rows: Sequence[Mapping[str, object]],
    columns: tuple[str, ...] | None = None,
    *,
    table_key: str | None = None,
) -> tuple[tuple[object, ...], ...]:
    """Convert mapping rows to a tuple of tuples in column order.

    Parameters
    ----------
    rows
        Rows as mapping objects.
    columns
        Column ordering to extract.
    table_key
        Table key used to resolve schema-based column order.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Tuple of row tuples in column order.

    Raises
    ------
    ValueError
        When neither table_key nor columns are provided.
    """
    if table_key is not None:
        serializer = row_serializer_for_table_key(table_key)
        return tuple(serializer(row) for row in rows)
    if columns is None:
        msg = "rows_to_tuples requires table_key or columns"
        raise ValueError(msg)
    return tuple(row_to_tuple(row, columns) for row in rows)


__all__ = ["row_to_tuple", "rows_to_tuples"]
