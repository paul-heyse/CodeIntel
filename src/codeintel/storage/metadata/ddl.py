"""DDL helpers for the DuckDB `metadata.*` schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.storage.constants import DUCKDB_DIALECT
from codeintel.storage.metadata.schema import METADATA_TABLES
from codeintel.storage.schema.sqlglot_ddl import (
    create_index_if_not_exists_ast,
    create_schema_if_not_exists_ast,
)
from codeintel.storage.schema_roundtrip import create_table_ast

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from codeintel.core.schemas.primitives import TableSchema

__all__ = ["apply_metadata_ddl"]


def apply_metadata_ddl(con: DuckDBPyConnection) -> None:
    """Create metadata schema tables required for runtime and export."""
    for table in METADATA_TABLES:
        _ensure_metadata_table(con, table)


def _ensure_metadata_table(con: DuckDBPyConnection, table: TableSchema) -> None:
    con.execute(create_schema_if_not_exists_ast(table.schema).sql(dialect=DUCKDB_DIALECT))
    con.execute(create_table_ast(table, if_not_exists=True).sql(dialect=DUCKDB_DIALECT))
    for index in table.indexes:
        index_sql = create_index_if_not_exists_ast(
            index_name=index.name,
            table_key=table.table_key,
            columns=index.columns,
            unique=index.unique,
        ).sql(dialect=DUCKDB_DIALECT)
        con.execute(index_sql)
