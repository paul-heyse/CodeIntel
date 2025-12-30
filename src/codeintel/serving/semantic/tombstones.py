"""SQLGlot tombstone filtering helpers for serving."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlglot import exp

from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    from collections.abc import Sequence


def apply_tombstone_filter(
    ast: exp.Select,
    *,
    table_key: str,
    primary_key: Sequence[str],
    snapshot_id: int | None,
) -> exp.Select:
    """Apply a NOT EXISTS tombstone filter to a simple SELECT.

    Returns
    -------
    sqlglot.expressions.Select
        Updated AST with tombstone filter applied when eligible.
    """
    if not primary_key:
        return ast
    if _has_joins(ast):
        return ast
    base_table = _base_table(ast)
    if base_table is None:
        return ast
    tombstone_table_key = _tombstone_table_key(table_key)
    if _contains_table(ast, tombstone_table_key):
        return ast
    base_alias = base_table.alias_or_name
    tombstone_expr, tombstone_alias = _tombstone_table_expr(tombstone_table_key)
    predicates: list[exp.Expression] = [
        exp.EQ(
            this=exp.column(col, table=base_alias),
            expression=exp.column(col, table=tombstone_alias),
        )
        for col in primary_key
    ]
    if snapshot_id is not None:
        predicates.append(
            exp.LTE(
                this=exp.column("snapshot_id", table=tombstone_alias),
                expression=exp.Literal.number(snapshot_id),
            )
        )
    if not predicates:
        return ast
    not_exists = exp.Not(
        this=exp.Exists(
            this=exp.select("1").from_(tombstone_expr).where(_combine_predicates(predicates))
        )
    )
    cloned = ast.copy()
    return cloned.where(not_exists)


def _base_table(ast: exp.Select) -> exp.Table | None:
    from_expr = ast.args.get("from_")
    if not isinstance(from_expr, exp.From):
        return None
    if ast.args.get("joins"):
        return None
    table_expr = from_expr.this
    if isinstance(table_expr, exp.Table):
        return table_expr
    return None


def _has_joins(ast: exp.Select) -> bool:
    return bool(ast.args.get("joins"))


def _tombstone_table_key(table_key: str) -> str:
    schema, table = split_table_key(table_key)
    return f"{schema}.{table}__tombstones"


def _tombstone_table_expr(table_key: str) -> tuple[exp.Table, str]:
    schema, table = split_table_key(table_key)
    table_expr = exp.Table(
        this=exp.to_identifier(table),
        db=exp.to_identifier(schema),
    )
    return table_expr, table


def _contains_table(ast: exp.Select, table_key: str) -> bool:
    schema, table = split_table_key(table_key)
    return any(node.name == table and node.db == schema for node in ast.find_all(exp.Table))


def _combine_predicates(predicates: Sequence[exp.Expression]) -> exp.Expression:
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined = exp.and_(combined, predicate)
    return combined


__all__ = ["apply_tombstone_filter"]
