"""Row conversion helpers for row-oriented materialization.

These helpers are intentionally **not** Hamilton nodes.
They are small, deterministic utilities that are safe to call inside compute
functions or tests without affecting Hamilton module discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def row_to_tuple(row: Mapping[str, object], columns: tuple[str, ...]) -> tuple[object, ...]:
    """Convert a mapping row to a tuple in column order.

    Parameters
    ----------
    row
        Row mapping.
    columns
        Column ordering to extract.

    Returns
    -------
    tuple[object, ...]
        Tuple of row values in column order. Missing keys yield None.
    """
    return tuple(row.get(col) for col in columns)


def rows_to_tuples(
    rows: Sequence[Mapping[str, object]],
    columns: tuple[str, ...],
) -> tuple[tuple[object, ...], ...]:
    """Convert mapping rows to a tuple of tuples in column order.

    Parameters
    ----------
    rows
        Rows as mapping objects.
    columns
        Column ordering to extract.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Tuple of row tuples in column order.
    """
    return tuple(row_to_tuple(row, columns) for row in rows)


__all__ = ["row_to_tuple", "rows_to_tuples"]
