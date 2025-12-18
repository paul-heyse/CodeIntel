"""SQLGlot DDL builders for DuckDB.

This module centralizes SQLGlot AST construction for common schema/index DDL
primitives used across storage subsystems (policy backend, metadata bootstrap,
schema automation). Keeping these builders in one place prevents semantic drift.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlglot.expressions as exp

__all__ = [
    "create_index_if_not_exists_ast",
    "create_schema_if_not_exists_ast",
]


def create_schema_if_not_exists_ast(schema_name: str) -> exp.Create:
    """Build a SQLGlot schema-create expression with IF NOT EXISTS semantics.

    Parameters
    ----------
    schema_name
        Schema name to create.

    Returns
    -------
    exp.Create
        SQLGlot expression for schema creation with IF NOT EXISTS.
    """
    return exp.Create(
        this=exp.to_identifier(schema_name),
        kind="SCHEMA",
        exists=True,
    )


def create_index_if_not_exists_ast(
    *,
    index_name: str,
    table_schema: str,
    table_name: str,
    columns: Sequence[str],
    unique: bool = False,
) -> exp.Create:
    """Build a SQLGlot index-create expression with IF NOT EXISTS semantics.

    Parameters
    ----------
    index_name
        Index name.
    table_schema
        Schema containing the indexed table.
    table_name
        Table name.
    columns
        Indexed column names, in order.
    unique
        When True, create a UNIQUE index.

    Returns
    -------
    exp.Create
        SQLGlot expression for index creation with IF NOT EXISTS.
    """
    table_expr = exp.Table(
        this=exp.to_identifier(table_name),
        db=exp.to_identifier(table_schema),
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
