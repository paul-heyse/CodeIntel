"""SQLGlot-based query builder for semantic specs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlglot import exp

from codeintel.core.schemas.type_mappings import normalize_engine_column_type
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.queries.filter_compiler import (
    FilterCompilerError,
    compile_filter_predicates,
    sqlglot_filter_expression,
)
from codeintel.storage.sqlglot_tools import (
    AstCapabilityConfig,
    canonicalize_select_duckdb,
    ensure_ast_capability,
    schema_mapping_for_table_key,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.core.schemas.primitives import ColumnType


class SqlglotQueryBuilderError(ValueError):
    """Raised when a SQLGlot query cannot be built."""


def build_sqlglot_query(
    *,
    spec: SemanticQuerySpec,
    allowed_anonymous_functions: frozenset[str] | None = None,
    allow_aggregates: bool = False,
    log_context: str = "serving_query_ast",
) -> exp.Select:
    """Build a SQLGlot expression for a semantic query spec.

    Parameters
    ----------
    spec
        Semantic query spec to translate into SQLGlot expressions.
    allowed_anonymous_functions
        Anonymous functions permitted in the query AST (None for default enforcement).
    allow_aggregates
        Whether aggregate functions are permitted in the AST.
    log_context
        Context label for AST capability warnings and logs.

    Returns
    -------
    sqlglot.expressions.Select
        SQLGlot Select expression representing the query.

    Raises
    ------
    SqlglotQueryBuilderError
        If pagination is invalid, columns are unknown, or filters cannot be compiled.
    """
    allowed_columns = spec.allowed_columns
    column_types = spec.column_types
    _validate_pagination(limit=spec.limit, offset=spec.offset)

    for col in spec.columns:
        _require_allowed_column(column=col, allowed_columns=allowed_columns, ctx="select")

    schema_name, table_name = split_table_key(spec.table_key)
    table_expr = exp.Table(
        this=exp.to_identifier(table_name),
        db=exp.to_identifier(schema_name),
    )

    select_exprs = [_projection_expr(col, column_types=column_types) for col in spec.columns]
    expr = exp.select(*select_exprs).from_(table_expr)

    if spec.filters:
        try:
            predicates = compile_filter_predicates(
                spec.filters,
                allowed_columns=allowed_columns,
                column_types=column_types,
            )
            predicate_expr = sqlglot_filter_expression(predicates)
        except FilterCompilerError as exc:
            raise SqlglotQueryBuilderError(str(exc)) from exc
        if predicate_expr is not None:
            expr = expr.where(predicate_expr)

    if spec.order_by:
        expr = expr.order_by(
            *_order_by_exprs(spec.order_by, allowed_columns=allowed_columns),
        )

    if spec.limit or spec.offset:
        expr = expr.limit(spec.limit)
        if spec.offset:
            expr = expr.offset(spec.offset)

    schema_mapping = schema_mapping_for_table_key(
        spec.table_key,
        column_types=column_types,
    )
    canonical = canonicalize_select_duckdb(expr, schema=schema_mapping)
    ensure_ast_capability(
        canonical,
        AstCapabilityConfig(
            allowed_anonymous_functions=allowed_anonymous_functions,
            allow_aggregates=allow_aggregates,
            log_context=log_context,
        ),
    )
    return canonical


def _validate_pagination(*, limit: int, offset: int) -> None:
    if limit < 0:
        msg = "limit must be >= 0"
        raise SqlglotQueryBuilderError(msg)
    if offset < 0:
        msg = "offset must be >= 0"
        raise SqlglotQueryBuilderError(msg)


def _require_allowed_column(*, column: str, allowed_columns: frozenset[str], ctx: str) -> None:
    if column not in allowed_columns:
        msg = f"Unknown {ctx} column: {column}"
        raise SqlglotQueryBuilderError(msg)


def _column_expr(column: str) -> exp.Column:
    return exp.Column(this=exp.to_identifier(column))


def _projection_expr(
    column: str,
    *,
    column_types: Mapping[str, ColumnType] | None,
) -> exp.Expression:
    col_expr = _column_expr(column)
    if column_types is None:
        return col_expr
    column_type = column_types.get(column)
    if column_type is None:
        return col_expr
    normalized = normalize_engine_column_type(column_type)
    if normalized is None:
        return col_expr
    try:
        data_type = exp.DataType.build(normalized, dialect="duckdb")
    except (TypeError, ValueError):
        return col_expr
    cast_expr = exp.Cast(this=col_expr, to=data_type)
    return exp.alias_(cast_expr, column)


def _order_by_exprs(
    order_by: Sequence[str],
    *,
    allowed_columns: frozenset[str],
) -> list[exp.Expression]:
    order_exprs: list[exp.Expression] = []
    for item in order_by:
        descending = item.startswith("-")
        col_name = item[1:] if descending else item
        _require_allowed_column(column=col_name, allowed_columns=allowed_columns, ctx="order_by")
        order_exprs.append(exp.Ordered(this=_column_expr(col_name), desc=descending))
    return order_exprs


__all__ = ["SqlglotQueryBuilderError", "build_sqlglot_query"]
