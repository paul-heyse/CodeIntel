"""Schema round-trip helpers (TableSchema → SQLGlot).

This module provides the bridge between contract schemas and SQLGlot DDL
generation without relying on Ibis.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import sqlglot
import sqlglot.expressions as exp
from sqlglot.errors import ParseError

from codeintel.storage.constants import DUCKDB_DIALECT

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(identifier: str, *, kind: str) -> str:
    if _IDENTIFIER_RE.fullmatch(identifier) is None:
        msg = f"Invalid {kind} identifier: {identifier!r}"
        raise ValueError(msg)
    return identifier


def _column_defs_sql(table: TableSchema) -> str:
    parts: list[str] = []
    for column in table.columns:
        _validate_identifier(column.name, kind="column")
        column_sql = f"{column.name} {column.type}"
        if not column.nullable:
            column_sql += " NOT NULL"
        parts.append(column_sql)

    if table.primary_key:
        for key in table.primary_key:
            _validate_identifier(key, kind="primary key column")
        pk_sql = f"PRIMARY KEY ({', '.join(table.primary_key)})"
        parts.append(pk_sql)

    return ", ".join(parts)


def create_table_ast(
    table: TableSchema,
    *,
    if_not_exists: bool,
    catalog: str | None = None,
) -> exp.Create:
    """Build a DuckDB table-creation DDL AST from `TableSchema`.

    Parameters
    ----------
    table
        Canonical schema contract.
    if_not_exists
        When True, includes an "IF NOT EXISTS" guard.
    catalog
        Optional catalog name to qualify the table.

    Returns
    -------
    sqlglot.expressions.Create
        SQLGlot AST for table creation.

    Raises
    ------
    ValueError
        If the generated DDL cannot be parsed or is invalid.
    TypeError
        If the generated DDL is not a CREATE statement.
    """
    _validate_identifier(table.schema, kind="schema")
    _validate_identifier(table.name, kind="table")
    qualifier = f"{table.schema}.{table.name}"
    if catalog is not None:
        _validate_identifier(catalog, kind="catalog")
        qualifier = f"{catalog}.{qualifier}"

    ddl_prefix = "CREATE TABLE IF NOT EXISTS" if if_not_exists else "CREATE TABLE"
    column_defs = _column_defs_sql(table)
    sql = f"{ddl_prefix} {qualifier} ({column_defs})"
    try:
        parsed = sqlglot.parse_one(sql, read=DUCKDB_DIALECT)
    except ParseError as exc:
        msg = "Failed to parse generated table DDL"
        raise ValueError(msg) from exc
    if not isinstance(parsed, exp.Create):
        msg = "Generated DDL did not produce a CREATE statement"
        raise TypeError(msg)
    return parsed


__all__ = ["create_table_ast"]
