"""DuckDB-backed schema inference for Ibis expressions.

This module implements the Phase 2 schema inference strategy:

- compile an Ibis expression to SQL
- run DuckDB ``DESCRIBE`` against that SQL
- map the resulting DuckDB types into the project TableSchema primitives

Notes
-----
Type normalization here is for *schema materialization*, not for hashing.
In particular, we preserve DECIMAL(38,0) rather than canonicalizing it to BIGINT.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import ibis

from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.storage.constants import DUCKDB_DIALECT
from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.storage.gateway.protocol import DuckDBConnection


_DECIMAL_RE = re.compile(r"^DECIMAL\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)$")
_DECIMAL_INT_PRECISION = 38
_DECIMAL_INT_SCALE = 0
_DESCRIBE_NULLABILITY_INDEX = 2


def _strip_trailing_semicolon(sql: str) -> str:
    """Remove a single trailing semicolon from SQL if present.

    Parameters
    ----------
    sql
        SQL query text.

    Returns
    -------
    str
        SQL query with at most one trailing semicolon removed.
    """
    return re.sub(r";\\s*$", "", sql.strip())


def normalize_duckdb_type(type_str: str) -> ColumnType:
    """Normalize a DuckDB type string into a supported ColumnType.

    Parameters
    ----------
    type_str
        Raw DuckDB type string from ``DESCRIBE`` output.

    Returns
    -------
    ColumnType
        Normalized type string compatible with TableSchema.

    Raises
    ------
    ValueError
        If the type cannot be normalized into a supported ColumnType.
    """
    normalized = " ".join(type_str.strip().upper().split())

    aliases: dict[str, ColumnType] = {
        "BOOL": "BOOLEAN",
        "BOOLEAN": "BOOLEAN",
        "INT": "INTEGER",
        "INTEGER": "INTEGER",
        "INT4": "INTEGER",
        "BIGINT": "BIGINT",
        "INT8": "BIGINT",
        "DOUBLE": "DOUBLE",
        "DOUBLE PRECISION": "DOUBLE",
        "VARCHAR": "VARCHAR",
        "TEXT": "VARCHAR",
        "JSON": "JSON",
        "TIMESTAMP": "TIMESTAMP",
        "TIMESTAMP_TZ": "TIMESTAMPTZ",
        "TIMESTAMPTZ": "TIMESTAMPTZ",
        "TIMESTAMP WITH TIME ZONE": "TIMESTAMPTZ",
        "DECIMAL": "DECIMAL",
        "DECIMAL(38,0)": "DECIMAL(38,0)",
    }
    if normalized in aliases:
        return aliases[normalized]

    decimal_match = _DECIMAL_RE.match(normalized)
    if decimal_match is not None:
        precision = int(decimal_match.group(1))
        scale = int(decimal_match.group(2))
        if precision == _DECIMAL_INT_PRECISION and scale == _DECIMAL_INT_SCALE:
            return "DECIMAL(38,0)"
        return "DECIMAL"

    msg = f"Unsupported DuckDB type for schema inference: {type_str!r}"
    raise ValueError(msg)


def infer_table_schema_from_sql(
    *,
    con: DuckDBConnection,
    sql: str,
    table_key: str,
) -> TableSchema:
    """Infer a TableSchema by running DuckDB ``DESCRIBE`` on a SQL query.

    Parameters
    ----------
    con
        DuckDB connection.
    sql
        SQL query to describe.
    table_key
        Table key (schema.table) to assign to the inferred schema.

    Returns
    -------
    TableSchema
        Inferred schema for the query output.
    """
    stripped_sql = _strip_trailing_semicolon(sql)
    rows = con.execute(f"DESCRIBE {stripped_sql}").fetchall()

    schema_name, table_name = split_table_key(table_key)

    columns: list[Column] = []
    for row in rows:
        col_name = str(row[0])
        col_type = normalize_duckdb_type(str(row[1]))
        nullable = True
        if len(row) > _DESCRIBE_NULLABILITY_INDEX:
            null_field = str(row[_DESCRIBE_NULLABILITY_INDEX]).strip().upper()
            if null_field in {"NO", "N", "FALSE", "0"}:
                nullable = False
            elif null_field in {"YES", "Y", "TRUE", "1"}:
                nullable = True
        columns.append(Column(name=col_name, type=col_type, nullable=nullable))

    return TableSchema(schema=schema_name, name=table_name, columns=columns)


def infer_table_schema_from_ibis(
    *,
    expr: ir.Table,
    con: DuckDBConnection,
    table_key: str,
) -> TableSchema:
    """Infer a TableSchema for an Ibis table expression using DuckDB DESCRIBE.

    Parameters
    ----------
    expr
        Ibis table expression to infer.
    con
        DuckDB connection used to run DESCRIBE.
    table_key
        Table key (schema.table) to assign to the inferred schema.

    Returns
    -------
    TableSchema
        Inferred schema for the Ibis expression output.
    """
    sql = ibis.to_sql(expr, dialect=DUCKDB_DIALECT)
    return infer_table_schema_from_sql(con=con, sql=sql, table_key=table_key)


__all__ = [
    "infer_table_schema_from_ibis",
    "infer_table_schema_from_sql",
    "normalize_duckdb_type",
]
