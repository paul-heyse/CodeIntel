"""DuckDB helpers for row operations and queries.

This module combines utilities for:
- Row count queries (from db_helpers)
- Bulk row insertion (from ingest_helpers)
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from duckdb import DuckDBPyConnection
from duckdb import Error as DuckDBError

from codeintel.config.datasets import get_dataset_contracts_by_table_key
from codeintel.storage.sql import build_insert_sql, quote_identifier

# Use DuckDBPyConnection directly to avoid circular imports with gateway
DuckDBConnection = DuckDBPyConnection

__all__ = [
    "DUCKDB_ERRORS",
    "macro_insert_rows",
    "row_counts_for_tables",
    "safe_row_counts",
]

DUCKDB_ERRORS: tuple[type[Exception], ...] = (DuckDBError,)


def row_counts_for_tables(
    con: DuckDBConnection,
    *,
    repo: str,
    commit: str,
    tables: Sequence[str],
) -> dict[str, int] | None:
    """Compute row counts for each table filtered by repo/commit.

    Returns
    -------
    dict[str, int] | None
        Mapping of table name to counts, or None if any table fails to count.
    """
    counts: dict[str, int] = {}
    for table in tables:
        try:
            escaped_repo = repo.replace("'", "''")
            escaped_commit = commit.replace("'", "''")
            relation = con.table(table).filter(
                f"repo = '{escaped_repo}' AND commit = '{escaped_commit}'"
            )
            result = relation.count("*").fetchone()
            if result is None:
                return None
            counts[table] = int(result[0])
        except DuckDBError:
            return None
    return counts


def safe_row_counts(
    con: DuckDBConnection | None,
    *,
    repo: str,
    commit: str,
    tables: Iterable[str],
) -> dict[str, int] | None:
    """Variant that tolerates missing connection or empty tables.

    Returns
    -------
    dict[str, int] | None
        Row counts or None when unavailable.
    """
    if con is None:
        return None
    return row_counts_for_tables(con, repo=repo, commit=commit, tables=tuple(tables))


def macro_insert_rows(
    con: DuckDBPyConnection,
    table_key: str,
    rows: Iterable[Sequence[object]],
) -> None:
    """Insert rows via ingest macro using table schema as ground truth.

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
    contract = get_dataset_contracts_by_table_key().get(table_key)
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
