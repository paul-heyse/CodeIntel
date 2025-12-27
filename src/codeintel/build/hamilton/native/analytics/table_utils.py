"""Shared helpers for analytics table construction."""

from __future__ import annotations

from codeintel.build.schemas import get_schema_provider
from codeintel.storage.gateway import DuckDBConnection, DuckDBRelation


def empty_relation_for_table(con: DuckDBConnection, table_key: str) -> DuckDBRelation:
    """Return an empty DuckDB relation matching the table schema.

    Parameters
    ----------
    con
        DuckDB connection used to build the relation.
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    DuckDBRelation
        Empty relation with the table's column names and types.
    """
    schema = get_schema_provider().require_table_schema(table_key)
    columns = [f"CAST(NULL AS {col.type}) AS {col.name}" for col in schema.columns]
    select_sql = f"SELECT {', '.join(columns)} WHERE 1=0"
    return con.sql(select_sql)


__all__ = ["empty_relation_for_table"]
