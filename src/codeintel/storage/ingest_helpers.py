"""Helpers for inserting bulk rows into DuckDB tables safely."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from duckdb import DuckDBPyConnection

__all__ = ["macro_insert_rows"]


def macro_insert_rows(
    con: DuckDBPyConnection,
    table_key: str,
    rows: Iterable[Sequence[object]],
) -> None:
    """
    Insert rows via ingest macro using table schema as ground truth.

    Pads missing trailing columns with NULLs; raises if rows exceed schema width.

    Raises
    ------
    ValueError
        If the table name is invalid or rows exceed the column count.
    """
    rows_list = list(rows)
    if not rows_list:
        return
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", table_key):
        message = f"Invalid table key: {table_key}"
        raise ValueError(message)
    schema_name, table_name = table_key.split(".", maxsplit=1)
    table_sql = f'"{schema_name}"."{table_name}"'
    pragma_sql = f"PRAGMA table_info({table_sql})"
    columns = [row[1] for row in con.execute(pragma_sql).fetchall()]
    if not columns:
        message = f"Table {table_key} missing"
        raise ValueError(message)
    col_count = len(columns)
    normalized: list[tuple[object, ...]] = []
    for row in rows_list:
        if len(row) > col_count:
            message = f"Row for {table_key} has {len(row)} values, exceeds column count {col_count}"
            raise ValueError(message)
        padded = tuple(row) + (None,) * (col_count - len(row))
        normalized.append(padded)
    view_name = f"temp_ingest_values_{table_name}"
    con.execute(f"DROP TABLE IF EXISTS {view_name}")
    con.table(table_key).limit(0).create(view_name)
    placeholders = ", ".join("?" for _ in columns)
    column_list = ", ".join(columns)
    con.executemany(
        f"INSERT INTO {view_name} ({column_list}) VALUES ({placeholders})",  # noqa: S608 - validated
        normalized,
    )
    con.table(view_name).insert_into(table_key)
    con.execute(f"DROP TABLE IF EXISTS {view_name}")
