"""Arrow plan builder for serving query subset translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from sqlglot import exp

from codeintel.core.columnar.expr_vocab import E, Expression
from codeintel.core.columnar.kernels import SortKey
from codeintel.core.columnar.queryspec import ProjectionSpec, QuerySpec
from codeintel.core.queries.filter_compiler import (
    FilterCompilerError,
    arrow_filter_expression,
    compile_filter_predicates,
)
from codeintel.core.schemas.type_mappings import arrow_type_from_column_type
from codeintel.serving.semantic.models import FilterSpec
from codeintel.serving.semantic.specs import SemanticQuerySpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.columnar.expr_vocab import Expression
    from codeintel.core.schemas.primitives import ColumnType


@dataclass(frozen=True, slots=True)
class ArrowPlanSpec:
    """Arrow plan details for the supported serving subset."""

    filter_expr: Expression | None
    projections: Mapping[str, Expression]
    order_by: tuple[SortKey, ...]
    limit: int | None


def build_arrow_plan_spec(
    *,
    spec: SemanticQuerySpec,
    ast: exp.Select,
) -> ArrowPlanSpec | None:
    """Build an Arrow plan spec when the AST stays within the supported subset.

    Returns
    -------
    ArrowPlanSpec | None
        Arrow plan spec when supported; otherwise None.
    """
    if spec.offset:
        return None
    if not _ast_supported(ast):
        return None
    projections = _projection_expressions(spec.columns, column_types=spec.column_types)
    if not projections:
        return None
    filter_expr = _filter_expression(
        spec.filters,
        allowed_columns=spec.allowed_columns,
        column_types=spec.column_types,
    )
    if spec.filters and filter_expr is None:
        return None
    order_by = _order_by(spec.order_by)
    limit = spec.limit if spec.limit > 0 else None
    return ArrowPlanSpec(
        filter_expr=filter_expr,
        projections=projections,
        order_by=order_by,
        limit=limit,
    )


def build_arrow_query_spec(plan_spec: ArrowPlanSpec) -> QuerySpec:
    """Return a QuerySpec representation for an Arrow plan.

    Returns
    -------
    QuerySpec
        Query specification for scan and plan compilation.
    """
    projection = ProjectionSpec(
        base_cols=tuple(plan_spec.projections.keys()),
        computed=tuple(plan_spec.projections.items()),
    )
    return QuerySpec(
        predicate=plan_spec.filter_expr,
        pushdown_predicate=plan_spec.filter_expr,
        projection=projection,
    )


def _ast_supported(ast: exp.Select) -> bool:
    if ast.args.get("joins"):
        return False
    if ast.args.get("group") or ast.args.get("having") or ast.args.get("qualify"):
        return False
    if ast.args.get("distinct"):
        return False
    if any(isinstance(node, exp.Star) for node in ast.find_all(exp.Star)):
        return False
    return _select_exprs_supported(ast.expressions)


def _select_exprs_supported(exprs: list[exp.Expression]) -> bool:
    if not exprs:
        return False
    return all(_select_expr_supported(expr) for expr in exprs)


def _select_expr_supported(expr: exp.Expression) -> bool:
    if isinstance(expr, exp.Column):
        return True
    if isinstance(expr, exp.Cast):
        inner = expr.this
        return isinstance(inner, exp.Column) if inner is not None else False
    if isinstance(expr, exp.Alias):
        alias = expr.alias
        if not alias:
            return False
        inner = expr.this
        if inner is None:
            return False
        return _select_expr_supported(inner)
    return False


def _projection_expressions(
    columns: list[str],
    *,
    column_types: Mapping[str, ColumnType] | None,
) -> dict[str, Expression]:
    projections: dict[str, Expression] = {}
    for column in columns:
        projections[column] = _projection_expr(column, column_types=column_types)
    return projections


def _projection_expr(
    column: str,
    *,
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    expr = E.field(column)
    if column_types is None:
        return expr
    column_type = column_types.get(column)
    if column_type is None:
        return expr
    try:
        arrow_type = arrow_type_from_column_type(column_type)
    except ValueError:
        return expr
    return E.cast(expr, arrow_type.to_string())


def _filter_expression(
    filters: list[FilterSpec],
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression | None:
    if not filters:
        return None
    try:
        predicates = compile_filter_predicates(
            filters,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    except FilterCompilerError:
        return None
    expression = arrow_filter_expression(predicates)
    if expression is None:
        return None
    return cast("Expression", expression)


def _order_by(order_by: list[str]) -> tuple[SortKey, ...]:
    keys: list[SortKey] = []
    for item in order_by:
        if not item:
            continue
        descending = item.startswith("-")
        column = item[1:] if descending else item
        keys.append((column, "descending" if descending else "ascending"))
    return tuple(keys)


__all__ = ["ArrowPlanSpec", "build_arrow_plan_spec", "build_arrow_query_spec"]
