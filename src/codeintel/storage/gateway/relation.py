"""Relation helpers for schema-qualified table or view access."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.storage.helpers.table_key import fully_qualified_table_ref

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection, DuckDBPyRelation


def relation_from_table_key(
    con: DuckDBPyConnection,
    table_key: str,
) -> DuckDBPyRelation:
    """Return a relation for a fully qualified table or view key.

    Parameters
    ----------
    con
        DuckDB connection used to build the relation.
    table_key
        Fully qualified table/view key (schema.table).

    Returns
    -------
    DuckDBPyRelation
        Relation bound to the requested table or view.
    """
    table_ref = fully_qualified_table_ref(table_key)
    return con.sql(f"SELECT * FROM {table_ref}")
