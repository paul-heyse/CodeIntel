"""DuckDB-backed schema inference for relation outputs.

This module implements the Phase 2 schema inference strategy:

- inspect DuckDB relation metadata
- map DuckDB types into the project TableSchema primitives

Notes
-----
Type normalization here is for *schema materialization*, not for hashing.
In particular, we preserve DECIMAL(38,0) rather than canonicalizing it to BIGINT.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.storage.gateway.protocol import DuckDBConnection, DuckDBRelation


_DECIMAL_RE = re.compile(r"^DECIMAL\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_DECIMAL_INT_PRECISION = 38
_DECIMAL_INT_SCALE = 0
_DESCRIBE_NULLABILITY_INDEX = 2
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def infer_table_schema_from_relation(
    *,
    relation: DuckDBRelation,
    table_key: str,
) -> TableSchema:
    """Infer a TableSchema for a DuckDB relation using relation metadata.

    Parameters
    ----------
    relation
        DuckDB relation to infer.
    table_key
        Table key (schema.table) to assign to the inferred schema.

    Returns
    -------
    TableSchema
        Inferred schema for the relation output.
    """
    schema_name, table_name = split_table_key(table_key)

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
    "infer_table_schema_from_relation",
    "infer_view_schema",
    "normalize_duckdb_type",
]
