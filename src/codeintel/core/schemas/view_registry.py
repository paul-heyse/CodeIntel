"""View schema derivation helpers for docs/graph view outputs."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from sqlglot import exp
from sqlglot.errors import SqlglotError

from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.sqlglot_tools import (
    canonicalize_expression_duckdb,
    extract_column_lineage_from_ast,
    schema_mapping_for_table_key,
)
from codeintel.core.views.discovery import discover_view_builders
from codeintel.core.views.inventory import view_builder_modules
from codeintel.core.views.protocol import ViewBuilder

LOG = logging.getLogger(__name__)

_VIEW_PREFIXES = ("docs.v_", "graph.v_")
_DEFAULT_COLUMN_TYPE = "VARCHAR"
_VIEW_SCHEMA_DESCRIPTION = "View schema derived from Hamilton view definitions."
_TABLE_KEY_PARTS = 2


def build_view_schema_overrides(
    base_schemas: Mapping[str, TableSchema],
) -> dict[str, TableSchema]:
    """Build TableSchema overrides for docs/graph views.

    Parameters
    ----------
    base_schemas
        Mapping of base table schemas used to infer view column types.

    Returns
    -------
    dict[str, TableSchema]
        View TableSchema overrides keyed by table_key.
    """
    base_column_map = _base_column_map(base_schemas)
    schema_mapping = _schema_mapping_for_base_schemas(base_schemas)
    base_schema_lookup = {
        table_key.lower(): schema
        for table_key, schema in base_schemas.items()
        if not table_key.startswith(_VIEW_PREFIXES)
    }
    overrides: dict[str, TableSchema] = {}
    for table_key, builder in _iter_view_builders():
        columns = _columns_from_view(
            table_key=table_key,
            builder=builder,
            base_columns=base_column_map,
            base_schemas=base_schema_lookup,
            schema_mapping=schema_mapping,
        )
        schema_name, table_name = _split_table_key(table_key)
        overrides[table_key] = TableSchema(
            schema=schema_name,
            name=table_name,
            columns=columns,
            description=_VIEW_SCHEMA_DESCRIPTION,
        )
    return overrides


def _iter_view_builders() -> Iterable[tuple[str, ViewBuilder]]:
    modules = view_builder_modules()
    if not modules:
        return ()
    try:
        builders = discover_view_builders(modules=modules)
    except ValueError:
        return ()
    return tuple(
        (discovered.table_key, discovered.builder)
        for discovered in builders
        if discovered.table_key.startswith(_VIEW_PREFIXES)
    )


def _schema_mapping_for_base_schemas(
    base_schemas: Mapping[str, TableSchema],
) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for table_key, schema in base_schemas.items():
        if table_key.startswith(_VIEW_PREFIXES):
            continue
        column_types = {col.name: col.type for col in schema.columns}
        normalized = schema_mapping_for_table_key(table_key, column_types=column_types)
        if normalized:
            mapping.update({key: dict(value) for key, value in normalized.items()})
    return mapping


def _base_column_map(base_schemas: Mapping[str, TableSchema]) -> dict[str, Column]:
    mapping: dict[str, Column] = {}
    for table_key, schema in base_schemas.items():
        if table_key.startswith(_VIEW_PREFIXES):
            continue
        table_prefix = table_key.lower()
        for column in schema.columns:
            mapping[f"{table_prefix}.{column.name.lower()}"] = column
    return mapping


def _columns_from_view(
    *,
    table_key: str,
    builder: ViewBuilder,
    base_columns: Mapping[str, Column],
    base_schemas: Mapping[str, TableSchema],
    schema_mapping: Mapping[str, Mapping[str, str]],
) -> list[Column]:
    ast = _run_view_builder(table_key=table_key, builder=builder)
    if ast is None:
        return []
    canonical = _canonicalize_view_ast(
        table_key=table_key,
        ast=ast,
        schema_mapping=schema_mapping,
    )
    if canonical is None:
        return []
    select_expr = _select_expression(canonical)
    if select_expr is None:
        LOG.warning("View %s did not resolve to a SELECT expression", table_key)
        return []
    lineage = _lineage_for_view(
        table_key=table_key,
        canonical=canonical,
        schema_mapping=schema_mapping,
    )
    context = _ColumnCollectionContext(
        select_expr=select_expr,
        alias_map=_alias_map(canonical),
        base_columns=base_columns,
        base_schemas=base_schemas,
        lineage=lineage,
    )
    return _columns_from_select(context)


def _run_view_builder(*, table_key: str, builder: ViewBuilder) -> exp.Expression | None:
    try:
        ast = builder()
    except (TypeError, ValueError) as exc:
        LOG.warning("View builder failed for %s: %s", table_key, exc)
        return None
    if not isinstance(ast, exp.Expression):
        LOG.warning("View builder for %s did not return a SQL expression", table_key)
        return None
    return ast


@dataclass(frozen=True, slots=True)
class _ColumnCollectionContext:
    select_expr: exp.Select
    alias_map: Mapping[str, str]
    base_columns: Mapping[str, Column]
    base_schemas: Mapping[str, TableSchema]
    lineage: Mapping[str, frozenset[str]]


def _canonicalize_view_ast(
    *,
    table_key: str,
    ast: exp.Expression,
    schema_mapping: Mapping[str, Mapping[str, str]],
) -> exp.Expression | None:
    try:
        return canonicalize_expression_duckdb(ast, schema=schema_mapping)
    except (SqlglotError, TypeError, ValueError) as exc:
        LOG.warning("View canonicalization failed for %s: %s", table_key, exc)
        return None


def _lineage_for_view(
    *,
    table_key: str,
    canonical: exp.Expression,
    schema_mapping: Mapping[str, Mapping[str, str]],
) -> Mapping[str, frozenset[str]]:
    try:
        return extract_column_lineage_from_ast(canonical, schema=schema_mapping)
    except SqlglotError as exc:
        LOG.warning("View lineage extraction failed for %s: %s", table_key, exc)
        return {}


def _columns_from_select(context: _ColumnCollectionContext) -> list[Column]:
    columns: list[Column] = []
    seen: set[str] = set()
    for expr in context.select_expr.expressions:
        if _is_star(expr):
            _extend_columns_from_star(
                expr=expr,
                context=context,
                columns=columns,
                seen=seen,
            )
            continue
        _append_expression_column(
            expr=expr,
            context=context,
            columns=columns,
            seen=seen,
        )
    return columns


def _extend_columns_from_star(
    *,
    expr: exp.Expression,
    context: _ColumnCollectionContext,
    columns: list[Column],
    seen: set[str],
) -> None:
    table_ref = _star_table(expr, context.alias_map) or _single_from_table(context.select_expr)
    if table_ref is None:
        return
    schema = context.base_schemas.get(table_ref.lower())
    if schema is None:
        return
    for column in schema.columns:
        if column.name in seen:
            continue
        columns.append(
            Column(
                name=column.name,
                type=column.type,
                nullable=column.nullable,
            )
        )
        seen.add(column.name)


def _append_expression_column(
    *,
    expr: exp.Expression,
    context: _ColumnCollectionContext,
    columns: list[Column],
    seen: set[str],
) -> None:
    name = expr.alias_or_name
    if not name or name in seen:
        return
    columns.append(
        _infer_column(name=name, lineage=context.lineage, base_columns=context.base_columns)
    )
    seen.add(name)


def _select_expression(ast: exp.Expression) -> exp.Select | None:
    root = ast.this if isinstance(ast, exp.Subquery) and ast.this is not None else ast
    if isinstance(root, exp.SetOperation) and root.this is not None:
        root = root.this
    return root if isinstance(root, exp.Select) else None


def _alias_map(ast: exp.Expression) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for table in ast.find_all(exp.Table):
        table_key = _table_key_for_table(table)
        alias = table.alias_or_name
        if alias:
            alias_map[alias.lower()] = table_key
        alias_map[table.name.lower()] = table_key
        if table.db:
            alias_map[f"{table.db}.{table.name}".lower()] = table_key
    return alias_map


def _table_key_for_table(table: exp.Table) -> str:
    schema = table.db
    name = table.name
    return f"{schema}.{name}" if schema else name


def _single_from_table(select_expr: exp.Select) -> str | None:
    from_expr = select_expr.args.get("from") or select_expr.args.get("from_")
    joins = select_expr.args.get("joins") or []
    if not isinstance(from_expr, exp.From) or joins:
        return None
    items = list(from_expr.expressions)
    if not items and from_expr.this is not None:
        items = [from_expr.this]
    if len(items) != 1:
        return None
    table = items[0]
    if not isinstance(table, exp.Table):
        return None
    return _table_key_for_table(table)


def _is_star(expr: exp.Expression) -> bool:
    return isinstance(expr, exp.Star) or (
        isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star)
    )


def _star_table(expr: exp.Expression, alias_map: Mapping[str, str]) -> str | None:
    if isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star):
        table = expr.table
        if table:
            return alias_map.get(table.lower(), table)
    return None


def _infer_column(
    *,
    name: str,
    lineage: Mapping[str, frozenset[str]],
    base_columns: Mapping[str, Column],
) -> Column:
    upstream = lineage.get(name) or lineage.get(name.lower())
    if not upstream:
        return Column(name=name, type=_DEFAULT_COLUMN_TYPE)
    candidates = [base_columns.get(ref) for ref in upstream if ref in base_columns]
    resolved = [column for column in candidates if column is not None]
    if not resolved:
        return Column(name=name, type=_DEFAULT_COLUMN_TYPE)
    types = {column.type for column in resolved}
    resolved_type = types.pop() if len(types) == 1 else _DEFAULT_COLUMN_TYPE
    nullable = any(column.nullable for column in resolved)
    return Column(name=name, type=resolved_type, nullable=nullable)


def _split_table_key(table_key: str) -> tuple[str, str]:
    parts = table_key.split(".", maxsplit=1)
    if len(parts) != _TABLE_KEY_PARTS or not parts[0] or not parts[1]:
        msg = f"Invalid table key: {table_key!r}"
        raise ValueError(msg)
    return parts[0], parts[1]


__all__ = ["build_view_schema_overrides"]
