"""AST-first SQLGlot query helpers for core utilities."""

from __future__ import annotations

from collections.abc import Mapping

from sqlglot import diff as sqlglot_diff
from sqlglot import exp

from codeintel.core.sqlglot_tools import canonicalize_expression_duckdb, parse_one_duckdb

SchemaMapping = Mapping[str, Mapping[str, str]]


def parse_ast(sql: str) -> exp.Expression:
    """Parse SQL into a SQLGlot AST using DuckDB dialect.

    Parameters
    ----------
    sql
        SQL text to parse.

    Returns
    -------
    sqlglot.exp.Expression
        Parsed SQLGlot AST.
    """
    return parse_one_duckdb(sql)


def canonicalize_ast(
    root: exp.Expression,
    *,
    schema: SchemaMapping | None = None,
) -> exp.Expression:
    """Canonicalize a SQLGlot AST with DuckDB settings.

    Parameters
    ----------
    root
        Root SQLGlot AST expression.
    schema
        Optional schema mapping used for normalization.

    Returns
    -------
    sqlglot.exp.Expression
        Canonicalized SQLGlot AST.
    """
    return canonicalize_expression_duckdb(root, schema=schema)


def coerce_ast(
    value: str | exp.Expression,
    *,
    schema: SchemaMapping | None = None,
) -> exp.Expression:
    """Coerce SQL text or AST into a canonical SQLGlot AST.

    Parameters
    ----------
    value
        SQL text or SQLGlot AST expression.
    schema
        Optional schema mapping used for normalization.

    Returns
    -------
    sqlglot.exp.Expression
        Canonicalized SQLGlot AST.
    """
    root = parse_one_duckdb(value) if isinstance(value, str) else value
    return canonicalize_expression_duckdb(root, schema=schema)


def diff_ast(before: exp.Expression, after: exp.Expression) -> tuple[str, ...]:
    """Return semantic diff actions between two SQLGlot ASTs.

    Parameters
    ----------
    before
        Baseline SQLGlot AST.
    after
        Updated SQLGlot AST.

    Returns
    -------
    tuple[str, ...]
        Human-readable semantic diff actions.
    """
    actions = sqlglot_diff(before, after)
    return tuple(str(action) for action in actions)


__all__ = [
    "canonicalize_ast",
    "coerce_ast",
    "diff_ast",
    "parse_ast",
]
