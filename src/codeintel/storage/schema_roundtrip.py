"""Schema round-trip helpers (TableSchema → SQLGlot).

This module provides the bridge between contract schemas and SQLGlot DDL
generation without relying on Ibis.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from codeintel.core.schemas.type_mappings import duckdb_pytype_from_column_type
from codeintel.storage.constants import DUCKDB_DIALECT

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(identifier: str, *, kind: str) -> str:
    if _IDENTIFIER_RE.fullmatch(identifier) is None:
        msg = f"Invalid {kind} identifier: {identifier!r}"
        raise ValueError(msg)
    return identifier


def _column_defs(table: TableSchema) -> list[exp.Expression]:
    expressions: list[exp.Expression] = []
    for column in table.columns:
        _validate_identifier(column.name, kind="column")
        duckdb_type = duckdb_pytype_from_column_type(column.type)
        type_sql = str(duckdb_type) if duckdb_type is not None else column.type
        try:
            data_type = exp.DataType.build(type_sql, dialect=DUCKDB_DIALECT)
        except (TypeError, ValueError) as exc:
            msg = f"Unsupported column type for DDL: {column.type!r}"
            raise ValueError(msg) from exc
        constraints: list[exp.ColumnConstraint] = []
        if not column.nullable:
            constraints.append(exp.ColumnConstraint(kind=exp.NotNullColumnConstraint()))
        expressions.append(
            exp.ColumnDef(
                this=exp.to_identifier(column.name),
                kind=data_type,
                constraints=constraints,
            )
        )

    if table.primary_key:
        for key in table.primary_key:
            _validate_identifier(key, kind="primary key column")
        pk_exprs = [
            exp.Ordered(this=exp.column(key), nulls_first=False) for key in table.primary_key
        ]
        expressions.append(
            exp.PrimaryKey(
                expressions=pk_exprs,
                include=exp.IndexParameters(),
            )
        )

    return expressions


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

    """
    _validate_identifier(table.schema, kind="schema")
    _validate_identifier(table.name, kind="table")
    table_expr = exp.Table(
        this=exp.to_identifier(table.name),
        db=exp.to_identifier(table.schema),
        catalog=exp.to_identifier(catalog) if catalog is not None else None,
    )
    if catalog is not None:
        _validate_identifier(catalog, kind="catalog")

    return exp.Create(
        this=exp.Schema(
            this=table_expr,
            expressions=_column_defs(table),
        ),
        kind="TABLE",
        exists=if_not_exists,
    )


__all__ = ["create_table_ast"]
