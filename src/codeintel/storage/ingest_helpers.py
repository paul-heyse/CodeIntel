"""Helpers for inserting bulk rows into DuckDB tables safely."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from duckdb import DuckDBPyConnection

from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.storage.sql_helpers import build_insert_sql, quote_identifier

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
        If the table has no DatasetContract schema.
    """
    rows_list = list(rows)
    if not rows_list:
        return
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", table_key):
        message = f"Invalid table key: {table_key}"
        raise ValueError(message)
    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
    if contract is None or contract.schema is None:
        message = f"Cannot insert into {table_key}: missing DatasetContract schema"
        raise ValueError(message)
    columns = [col.name for col in contract.schema.columns if col.name is not None]
    _, table_name = table_key.split(".", maxsplit=1)
    col_count = len(columns)
    normalized: list[tuple[object, ...]] = []
    for row in rows_list:
        if len(row) > col_count:
            message = f"Row for {table_key} has {len(row)} values, exceeds column count {col_count}"
            raise ValueError(message)
        padded = tuple(row) + (None,) * (col_count - len(row))
        normalized.append(padded)
    view_name = f"temp_ingest_values_{table_name}"
    view_sql = quote_identifier(view_name)
    con.execute(f"DROP TABLE IF EXISTS {view_sql}")
    con.table(table_key).limit(0).create(view_sql)
    insert_sql = build_insert_sql(
        view_sql,
        columns,
        identifier_is_quoted=True,
    )
    con.executemany(insert_sql, normalized)
    con.table(view_name).insert_into(table_key)
    con.execute(f"DROP TABLE IF EXISTS {view_sql}")
