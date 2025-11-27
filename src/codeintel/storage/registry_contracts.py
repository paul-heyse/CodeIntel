"""Helpers to derive schema contracts from the live DuckDB registry/tables.

These utilities read the tables created by metadata_bootstrap (e.g., the
schema registry and the actual table definitions in the catalog) and expose
lightweight contracts that can be used to generate ingestion DDL or column
lists without relying on the legacy Python schema definitions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, NamedTuple

import duckdb

if TYPE_CHECKING:
    from codeintel.storage.gateway import DuckDBConnection
else:
    DuckDBConnection = duckdb.DuckDBPyConnection


class ColumnDef(NamedTuple):
    """Column name and DuckDB-rendered data type."""

    name: str
    data_type: str


def _split_table_key(table_key: str) -> tuple[str, str]:
    if "." not in table_key:
        message = f"Invalid table key (expected schema.table): {table_key}"
        raise ValueError(message)
    schema, name = table_key.split(".", maxsplit=1)
    return schema, name


def list_registry_tables(con: DuckDBConnection) -> list[str]:
    """
    Return all table keys tracked in metadata.dataset_schema_registry.

    Returns
    -------
    list[str]
        Ordered table keys present in the registry.
    """
    rows = con.execute(
        "SELECT table_key FROM metadata.dataset_schema_registry ORDER BY table_key"
    ).fetchall()
    return [str(row[0]) for row in rows]


def fetch_table_columns(con: DuckDBConnection, table_key: str) -> list[ColumnDef]:
    """
    Read column definitions for a table from the catalog.

    Parameters
    ----------
    con :
        Active DuckDB connection.
    table_key : str
        Fully qualified table name (schema.table).

    Returns
    -------
    list[ColumnDef]
        Column metadata in ordinal order.
    """
    schema, name = _split_table_key(table_key)
    rows = con.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ?
        ORDER BY ordinal_position
        """,
        [schema, name],
    ).fetchall()
    return [ColumnDef(str(col), str(dtype)) for col, dtype in rows]


def build_registry_contracts(
    con: DuckDBConnection, *, table_keys: Iterable[str] | None = None
) -> Mapping[str, list[ColumnDef]]:
    """
    Build a mapping of table key -> column definitions from the catalog.

    Parameters
    ----------
    con :
        Active DuckDB connection.
    table_keys : Iterable[str] | None, optional
        Subset of table keys to inspect; defaults to all registry entries.

    Returns
    -------
    Mapping[str, list[ColumnDef]]
        Contracts derived from the live database definitions.
    """
    keys = list(table_keys) if table_keys is not None else list_registry_tables(con)
    return {table_key: fetch_table_columns(con, table_key) for table_key in keys}


def render_create_table_from_catalog(con: DuckDBConnection, table_key: str) -> str:
    """
    Render a simple CREATE TABLE statement from catalog column metadata.

    Parameters
    ----------
    con :
        Active DuckDB connection.
    table_key : str
        Fully qualified table name (schema.table).

    Returns
    -------
    str
        Deterministic CREATE TABLE DDL reflecting the current catalog types.
    """
    cols = fetch_table_columns(con, table_key)
    column_lines = ",\n  ".join(f"{col.name} {col.data_type}" for col in cols)
    return f"CREATE TABLE IF NOT EXISTS {table_key} (\n  {column_lines}\n);"
