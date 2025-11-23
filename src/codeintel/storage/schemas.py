"""
DuckDB schema definitions for the CodeIntel metadata warehouse.

These DDLs are derived from README_METADATA.md ("CodeIntel Metadata Outputs")
and cover all exported datasets (goids, call graph, CFG/DFG, coverage, tests,
risk factors, etc.).
"""

from __future__ import annotations

from collections.abc import Iterable

from duckdb import DuckDBPyConnection

from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema

SCHEMAS = ("core", "graph", "analytics", "docs")


def _quote(identifier: str) -> str:
    """
    Quote an identifier for DuckDB.

    Returns
    -------
    str
        Identifier wrapped in double quotes with internal quotes escaped.
    """
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _build_table_ddl(table: TableSchema) -> str:
    """
    Generate CREATE TABLE DDL from a TableSchema.

    Returns
    -------
    str
        CREATE TABLE statement for the provided schema.
    """
    col_lines: list[str] = []
    for col in table.columns:
        nullable_sql = "" if col.nullable else " NOT NULL"
        col_lines.append(f"    {_quote(col.name)} {col.type}{nullable_sql}")
    if table.primary_key:
        pk_cols = ", ".join(_quote(col) for col in table.primary_key)
        col_lines.append(f"    PRIMARY KEY ({pk_cols})")
    cols_sql = ",\n".join(col_lines)
    return (
        f"DROP TABLE IF EXISTS {_quote(table.schema)}.{_quote(table.name)};\n"
        f"CREATE TABLE {_quote(table.schema)}.{_quote(table.name)} (\n"
        f"{cols_sql}\n"
        ");"
    )


TABLE_DDL: dict[str, str] = {key: _build_table_ddl(schema) for key, schema in TABLE_SCHEMAS.items()}


def _build_index_ddl(table: TableSchema) -> list[str]:
    statements: list[str] = []
    for index in table.indexes:
        columns = ", ".join(_quote(col) for col in index.columns)
        uniqueness = "UNIQUE " if index.unique else ""
        statements.append(
            f"CREATE {uniqueness}INDEX IF NOT EXISTS {_quote(index.name)} "
            f"ON {_quote(table.schema)}.{_quote(table.name)}({columns});"
        )
    return statements


INDEX_DDL: tuple[str, ...] = tuple(
    ddl for schema in TABLE_SCHEMAS.values() for ddl in _build_index_ddl(schema)
)


def create_schemas(con: DuckDBPyConnection) -> None:
    """Ensure logical schemas (core, graph, analytics, docs) exist."""
    for schema in SCHEMAS:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")


def apply_all_schemas(
    con: DuckDBPyConnection,
    extra_ddl: Iterable[str] | None = None,
) -> None:
    """
    Create all known tables in the current DuckDB database.

    Call this once at startup before running any pipeline steps that
    insert into these tables.
    """
    create_schemas(con)

    for ddl in TABLE_DDL.values():
        con.execute(ddl)

    for ddl in INDEX_DDL:
        con.execute(ddl)

    if extra_ddl:
        for stmt in extra_ddl:
            con.execute(stmt)
