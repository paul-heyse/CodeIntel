"""DuckDB-backed schema inference for Ibis expressions.

This module implements the Phase 2 schema inference strategy:

- compile an Ibis expression to SQL
- use DuckDB's relation metadata for schema
- map DuckDB types into the project TableSchema primitives

Notes
-----
Type normalization here is for *schema materialization*, not for hashing.
In particular, we preserve DECIMAL(38,0) rather than canonicalizing it to BIGINT.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import ibis
import sqlglot
import sqlglot.expressions as exp
from sqlglot.errors import ParseError

from codeintel.build.table_keys import split_table_key
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.storage.constants import DUCKDB_DIALECT

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.storage.gateway.protocol import DuckDBConnection


_DECIMAL_RE = re.compile(r"^DECIMAL\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_DECIMAL_INT_PRECISION = 38
_DECIMAL_INT_SCALE = 0
_DESCRIBE_NULLABILITY_INDEX = 2
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_BANNED_SQLGLOT_NODES: tuple[type[exp.Expression], ...] = (
    exp.Alter,
    exp.Command,
    exp.Create,
    exp.Delete,
    exp.Drop,
    exp.Insert,
    exp.Transaction,
    exp.TruncateTable,
    exp.Update,
)


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
    return re.sub(r";\s*$", "", sql.strip())


def _validate_identifier(identifier: str, *, kind: str) -> str:
    """Validate a DuckDB identifier.

    Parameters
    ----------
    identifier
        Identifier string to validate.
    kind
        Human-friendly kind used in error messages (e.g., "schema", "table").

    Returns
    -------
    str
        The validated identifier.

    Raises
    ------
    ValueError
        If the identifier does not match the allowed pattern.
    """
    if _IDENTIFIER_RE.fullmatch(identifier) is None:
        msg = f"Invalid DuckDB identifier for {kind}: {identifier!r}"
        raise ValueError(msg)
    return identifier


def _validate_trusted_select_sql(sql: str) -> str:
    """Validate a SQL string is a single SELECT-like query with no DDL/DML.

    Parameters
    ----------
    sql
        SQL query text.

    Returns
    -------
    str
        Stripped SQL query text safe to pass to ``DuckDBConnection.sql``.

    Raises
    ------
    ValueError
        If parsing fails or the query contains DDL/DML statements.
    """
    stripped_sql = _strip_trailing_semicolon(sql)
    try:
        parsed = sqlglot.parse_one(stripped_sql, read=DUCKDB_DIALECT)
    except ParseError as exc:
        msg = "Failed to parse SQL for schema inference"
        raise ValueError(msg) from exc

    for node in parsed.walk():
        if type(node) in _BANNED_SQLGLOT_NODES:
            msg = "Schema inference only supports SELECT-style queries (DDL/DML is not allowed)"
            raise ValueError(msg)

    return stripped_sql


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
        "HUGEINT": "DECIMAL(38,0)",
        "UHUGEINT": "DECIMAL(38,0)",
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
    validated_sql = _validate_trusted_select_sql(sql)

    schema_name, table_name = split_table_key(table_key)
    relation = con.sql(validated_sql)

    columns: list[Column] = []
    for col_name, dtype in zip(relation.columns, relation.types, strict=True):
        col_type = normalize_duckdb_type(str(dtype))
        columns.append(Column(name=str(col_name), type=col_type, nullable=True))

    return TableSchema(schema=schema_name, name=table_name, columns=columns)


def infer_view_schema(
    *,
    con: DuckDBConnection,
    view_key: str,
) -> TableSchema:
    """Infer a TableSchema for an existing DuckDB view.

    Use this to infer schemas for views that have already been created in
    the database. The function uses ``DESCRIBE schema.view_name`` to retrieve
    column information.

    Parameters
    ----------
    con
        DuckDB connection with the view defined.
    view_key
        Fully qualified view key (schema.view_name).

    Returns
    -------
    TableSchema
        Inferred schema for the view.

    Examples
    --------
    >>> schema = infer_view_schema(con=con, view_key="docs.v_function_summary")
    >>> schema.table_key
    'docs.v_function_summary'
    """
    schema_name, view_name = split_table_key(view_key)
    _validate_identifier(schema_name, kind="schema")
    _validate_identifier(view_name, kind="table/view")

    # Use DESCRIBE because it preserves NOT NULL constraints for many views, and
    # allows duckdb.CatalogException to surface for nonexistent objects.
    rows = con.execute(f"DESCRIBE {schema_name}.{view_name}").fetchall()

    columns: list[Column] = []
    for row in rows:
        col_name = str(row[0])
        col_type = normalize_duckdb_type(str(row[1]))
        nullable = True
        nullable_raw = (
            str(row[_DESCRIBE_NULLABILITY_INDEX]).strip().upper()
            if len(row) > _DESCRIBE_NULLABILITY_INDEX
            else "YES"
        )
        if nullable_raw in {"NO", "N", "FALSE", "0"}:
            nullable = False
        columns.append(Column(name=col_name, type=col_type, nullable=nullable))

    return TableSchema(schema=schema_name, name=view_name, columns=columns)


__all__ = [
    "infer_table_schema_from_ibis",
    "infer_view_schema",
    "normalize_duckdb_type",
]
