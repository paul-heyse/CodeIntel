"""SQLGlot DDL builders for DuckDB schema and index operations."""

from __future__ import annotations

import re
from collections.abc import Sequence

import sqlglot.expressions as exp

from codeintel.storage.helpers.table_key import split_table_key

__all__ = [
    "create_index_if_not_exists_ast",
    "create_schema_if_not_exists_ast",
]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(identifier: str, *, kind: str) -> str:
    if _IDENTIFIER_RE.fullmatch(identifier) is None:
        msg = f"Invalid {kind} identifier: {identifier!r}"
        raise ValueError(msg)
    return identifier


def create_schema_if_not_exists_ast(
    schema_name: str,
    *,
    catalog: str | None = None,
) -> exp.Create:
    """Build a SQLGlot schema-create expression with IF NOT EXISTS semantics.

    Parameters
    ----------
    schema_name
        Schema name to create.
    catalog
        Optional catalog name to qualify the schema.

    Returns
    -------
    exp.Create
        SQLGlot expression for schema creation with IF NOT EXISTS.

    Raises
    ------
    ValueError
        If identifiers are invalid.
    """
    if _IDENTIFIER_RE.fullmatch(schema_name) is None:
        msg = f"Invalid schema identifier: {schema_name!r}"
        raise ValueError(msg)
    if catalog is not None and _IDENTIFIER_RE.fullmatch(catalog) is None:
        msg = f"Invalid catalog identifier: {catalog!r}"
        raise ValueError(msg)
    qualifier = exp.Table(
        this=exp.to_identifier(schema_name),
        db=exp.to_identifier(catalog) if catalog is not None else None,
    )

    return exp.Create(
        this=qualifier,
        kind="SCHEMA",
        exists=True,
    )


def create_index_if_not_exists_ast(
    *,
    index_name: str,
    table_key: str,
    columns: Sequence[str],
    unique: bool = False,
    catalog: str | None = None,
) -> exp.Create:
    """Build a SQLGlot index-create expression with IF NOT EXISTS semantics.

    Parameters
    ----------
    index_name
        Index name.
    table_key
        Schema-qualified table key (e.g., "core.modules").
    columns
        Indexed column names, in order.
    unique
        When True, create a UNIQUE index.
    catalog
        Optional catalog name to qualify the table.

    Returns
    -------
    exp.Create
        SQLGlot expression for index creation with IF NOT EXISTS.
    """
    _validate_identifier(index_name, kind="index")
    table_schema, table_name = split_table_key(table_key)
    _validate_identifier(table_schema, kind="schema")
    _validate_identifier(table_name, kind="table")
    for column in columns:
        _validate_identifier(column, kind="column")
    if catalog is not None:
        _validate_identifier(catalog, kind="catalog")
    table_expr = exp.Table(
        this=exp.to_identifier(table_name),
        db=exp.to_identifier(table_schema),
        catalog=exp.to_identifier(catalog) if catalog is not None else None,
    )

    index_columns = [exp.Ordered(this=exp.Column(this=exp.to_identifier(col))) for col in columns]
    index_params = exp.IndexParameters(columns=index_columns)
    index_expr = exp.Index(
        this=exp.to_identifier(index_name),
        table=table_expr,
        params=index_params,
    )

    return exp.Create(
        this=index_expr,
        kind="INDEX",
        exists=True,
        unique=unique,
    )
