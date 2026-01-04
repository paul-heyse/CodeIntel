"""Arrow reader/table helpers for graph assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa

from codeintel.build.tabular.compute_columns import empty_table
from codeintel.build.tabular.conversion import reader_to_table, table_to_reader
from codeintel.build.tabular.conversion import tabular_to_arrow_reader as tabular_to_reader


def table_rows(table: pa.Table) -> list[dict[str, object]]:
    """Return table rows as a list of dictionaries.

    Returns
    -------
    list[dict[str, object]]
        Row mappings for the table.
    """
    if table.num_rows == 0:
        return []
    return table.to_pylist()


def select_table_columns(table: pa.Table, columns: Sequence[str]) -> pa.Table:
    """Select a subset of columns, returning an empty table if none exist.

    Returns
    -------
    pyarrow.Table
        Table with selected columns.
    """
    if not columns:
        return table
    present = [column for column in columns if column in table.column_names]
    if not present:
        return empty_table(columns)
    return table.select(present)


def ensure_table_columns(table: pa.Table, columns: Sequence[str]) -> pa.Table:
    """Ensure a table has the requested columns by appending nulls as needed.

    Returns
    -------
    pyarrow.Table
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

    Returns
    -------
    pyarrow.Table
        Table without dropped columns.
    """
    if not columns:
        return table
    existing = [column for column in table.column_names if column not in set(columns)]
    if not existing:
        return empty_table([])
    return table.select(existing)


def rename_table_columns(table: pa.Table, mapping: Mapping[str, str]) -> pa.Table:
    """Rename table columns using the provided mapping.

    Returns
    -------
    pyarrow.Table
        Table with renamed columns.
    """
    if not mapping:
        return table
    names = [mapping.get(name, name) for name in table.column_names]
    return table.rename_columns(names)


__all__ = [
    "drop_table_columns",
    "ensure_table_columns",
    "reader_to_table",
    "rename_table_columns",
    "select_table_columns",
    "table_rows",
    "table_to_reader",
    "tabular_to_reader",
]
