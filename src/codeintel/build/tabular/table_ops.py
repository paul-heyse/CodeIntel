"""Arrow table helpers for common column operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import polars as pl
import pyarrow as pa

from codeintel.build.tabular.compute_columns import empty_table
from codeintel.core.columnar.iter import iter_rows


def empty_table_for_columns(columns: Sequence[str]) -> pa.Table:
    """Return an empty table with null-typed columns.

    Parameters
    ----------
    columns
        Column names to include in the empty table.

    Returns
    -------
    table : pa.Table
        Empty Arrow table with null-typed columns.
    """
    return empty_table(columns)


def select_table_columns(table: pa.Table, columns: Sequence[str]) -> pa.Table:
    """Select a subset of columns, returning an empty table if none exist.

    Parameters
    ----------
    table
        Source Arrow table.
    columns
        Column names to select.

    Returns
    -------
    selected : pa.Table
        Table with selected columns.
    """
    if not columns:
        return table
    present = [column for column in columns if column in table.column_names]
    if not present:
        return empty_table_for_columns(columns)
    return table.select(present)


def ensure_table_columns(table: pa.Table, columns: Sequence[str]) -> pa.Table:
    """Ensure a table has the requested columns by appending nulls as needed.

    Parameters
    ----------
    table
        Source Arrow table.
    columns
        Column names to ensure.

    Returns
    -------
    ensured : pa.Table
        Table with requested columns ensured.
    """
    if not columns:
        return table
    existing = set(table.column_names)
    arrays = [
        table[column] if column in existing else pa.nulls(table.num_rows) for column in columns
    ]
    return pa.Table.from_arrays(arrays, names=list(columns))


def drop_table_columns(table: pa.Table, columns: Sequence[str]) -> pa.Table:
    """Drop columns if they exist.

    Parameters
    ----------
    table
        Source Arrow table.
    columns
        Column names to drop.

    Returns
    -------
    filtered : pa.Table
        Table without dropped columns.
    """
    if not columns:
        return table
    existing = [column for column in table.column_names if column not in set(columns)]
    if not existing:
        return table.select([])
    return table.select(existing)


def rename_table_columns(table: pa.Table, mapping: Mapping[str, str]) -> pa.Table:
    """Rename table columns using the provided mapping.

    Parameters
    ----------
    table
        Source Arrow table.
    mapping
        Column rename mapping.

    Returns
    -------
    renamed : pa.Table
        Table with renamed columns.
    """
    if not mapping:
        return table
    names = [mapping.get(name, name) for name in table.column_names]
    return table.rename_columns(names)


def table_rows(table: pa.Table) -> list[dict[str, object]]:
    """Return table rows as a list of dictionaries.

    Parameters
    ----------
    table
        Source Arrow table.

    Returns
    -------
    rows : list[dict[str, object]]
        Row mappings for the table.
    """
    if table.num_rows == 0:
        return []
    return list(iter_rows(table))


def to_records(frame: pa.Table | pl.DataFrame | pl.LazyFrame) -> list[dict[str, object]]:
    """Convert a columnar frame into a list of dictionaries.

    Parameters
    ----------
    frame
        Arrow table or Polars frame to convert.

    Returns
    -------
    records : list[dict[str, object]]
        Row dictionaries converted from the input frame.
    """
    if isinstance(frame, pa.Table):
        return table_rows(frame)
    if isinstance(frame, pl.LazyFrame):
        return cast("list[dict[str, object]]", frame.collect().to_dicts())
    return cast("list[dict[str, object]]", frame.to_dicts())


__all__ = [
    "drop_table_columns",
    "empty_table_for_columns",
    "ensure_table_columns",
    "rename_table_columns",
    "select_table_columns",
    "table_rows",
    "to_records",
]
