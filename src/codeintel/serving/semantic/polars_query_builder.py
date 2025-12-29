"""Polars query builder for semantic specs."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from sqlglot import exp

from codeintel.core.schemas.primitives import column_type_base
from codeintel.serving.semantic.filter_ops import allowed_ops_for_column_type

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from polars import Expr, LazyFrame

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.semantic.models import FilterScalar

    type PolarsExpr = Expr
    type PolarsLazyFrame = LazyFrame
else:
    type PolarsExpr = object
    type PolarsLazyFrame = object

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


class PolarsQueryBuilderError(ValueError):
    """Raised when Polars query construction fails."""


class _PolarsApi(Protocol):
    def col(self, name: str) -> PolarsExpr: ...

    def lit(self, value: object) -> PolarsExpr: ...


def _require_polars() -> _PolarsApi:
    if pl is None:  # pragma: no cover
        msg = "polars is required for Polars query execution"
        raise PolarsQueryBuilderError(msg)
    return cast("_PolarsApi", pl)


def can_apply_query_ast(
    *,
    ast: exp.Expression,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None = None,
) -> bool:
    """Return True when the SQLGlot AST can be applied by the Polars builder.

    Parameters
    ----------
    ast
        SQLGlot Select expression to validate.
    allowed_columns
        Columns permitted for selection, filtering, and ordering.
    column_types
        Optional column type mapping for operator validation.

    Returns
    -------
    bool
        True when the AST can be satisfied without builder errors.
    """
    if pl is None:  # pragma: no cover
        return False
    try:
        _parse_ast_components(
            ast=ast,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    except PolarsQueryBuilderError:
        return False
    return True


def apply_query_ast(
    lazyframe: PolarsLazyFrame,
    *,
    ast: exp.Expression,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None = None,
) -> PolarsLazyFrame:
    """Apply a SQLGlot AST to a Polars LazyFrame.

    Parameters
    ----------
    lazyframe
        Input Polars LazyFrame to filter and project.
    ast
        SQLGlot Select expression to translate.
    allowed_columns
        Columns permitted for selection, filtering, and ordering.
    column_types
        Optional column type mapping for operator validation.

    Returns
    -------
    polars.LazyFrame
        The filtered, ordered, and sliced lazy frame.

    Raises
    ------
    PolarsQueryBuilderError
        If the AST is invalid or uses unsupported operators.
    """
    if pl is None:  # pragma: no cover
        msg = "polars is required for Polars query execution"
        raise PolarsQueryBuilderError(msg)
    components = _parse_ast_components(
        ast=ast,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    if components.filters is not None:
        lazyframe = lazyframe.filter(components.filters)
    if components.order_by:
        lazyframe = lazyframe.sort(by=components.order_by, descending=components.descending)
    lazyframe = lazyframe.select(components.select_exprs)
    if components.limit is not None or components.offset:
        if components.limit is None:
            msg = "offset without limit is not supported"
            raise PolarsQueryBuilderError(msg)
        lazyframe = lazyframe.slice(components.offset, components.limit)
    return lazyframe


def _validate_pagination(*, limit: int, offset: int) -> None:
    if limit < 0:
        msg = "limit must be >= 0"
        raise PolarsQueryBuilderError(msg)
    if offset < 0:
        msg = "offset must be >= 0"
        raise PolarsQueryBuilderError(msg)


def _require_allowed_column(*, column: str, allowed_columns: frozenset[str], ctx: str) -> None:
    if column not in allowed_columns:
        msg = f"Unknown {ctx} column: {column}"
        raise PolarsQueryBuilderError(msg)


@dataclass(frozen=True, slots=True)
class _AstQueryComponents:
    select_exprs: list[PolarsExpr]
    filters: PolarsExpr | None
    order_by: list[str]
    descending: list[bool]
    limit: int | None
    offset: int


def _parse_ast_components(
    *,
    ast: exp.Expression,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> _AstQueryComponents:
    if not isinstance(ast, exp.Select):
        msg = "Expected SQLGlot Select expression"
        raise PolarsQueryBuilderError(msg)
    select_exprs = _select_exprs_from_ast(
        ast,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    filters = _where_expr_from_ast(
        ast=ast,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    order_by, descending = _order_by_from_ast(ast, allowed_columns=allowed_columns)
    limit, offset = _limit_offset_from_ast(ast)
    _validate_pagination(limit=limit or 0, offset=offset)
    return _AstQueryComponents(
        select_exprs=select_exprs,
        filters=filters,
        order_by=order_by,
        descending=descending,
        limit=limit,
        offset=offset,
    )


def _select_exprs_from_ast(
    ast: exp.Select,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> list[PolarsExpr]:
    select_exprs = [
        _polars_expr_from_select(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        for expr in ast.expressions
    ]
    if not select_exprs:
        msg = "Select expression must include at least one column"
        raise PolarsQueryBuilderError(msg)
    return select_exprs


def _polars_expr_from_select(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    if isinstance(expr, exp.Alias):
        alias = expr.alias
        if not alias:
            msg = "Select alias requires a name"
            raise PolarsQueryBuilderError(msg)
        base = _polars_expr_from_select(
            expr.this,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        return base.alias(alias)
    return _polars_expr_from_projection(
        expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _polars_expr_from_projection(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    result: PolarsExpr | None = None
    if isinstance(expr, exp.Column):
        column = _column_name(expr)
        _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="select")
        pl_mod = _require_polars()
        result = pl_mod.col(column)
    elif isinstance(expr, exp.Lower):
        result = _polars_string_unary_expr(
            expr.this,
            func_name="lower",
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, exp.Upper):
        result = _polars_string_unary_expr(
            expr.this,
            func_name="upper",
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, exp.Coalesce):
        result = _polars_coalesce_expr(
            expr.expressions,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, exp.Contains):
        result = _polars_string_predicate_expr(
            expr.this,
            expr.expression,
            func_name="contains",
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, exp.StartsWith):
        result = _polars_string_predicate_expr(
            expr.this,
            expr.expression,
            func_name="starts_with",
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, exp.Anonymous):
        result = _polars_function_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, (exp.Boolean, exp.Literal)):
        result = _polars_literal_expr(expr)

    if result is None:
        msg = f"Unsupported select expression: {type(expr).__name__}"
        raise PolarsQueryBuilderError(msg)
    return result


def _polars_literal_expr(expr: exp.Expression) -> PolarsExpr:
    pl_mod = _require_polars()
    value = _literal_value(expr)
    return pl_mod.lit(value)


def _polars_function_expr(
    expr: exp.Anonymous,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    func_name = (expr.name or "").lower()
    if func_name in _STRING_PREDICATE_FUNCS:
        return _polars_string_function(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if func_name in _STRING_UNARY_FUNCS:
        column_expr = expr.expressions[0] if expr.expressions else None
        return _polars_string_unary_expr(
            column_expr,
            func_name=func_name,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if func_name in _CASE_FUNCTIONS:
        return _polars_coalesce_expr(
            expr.expressions,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    msg = f"Unsupported function: {func_name or '<unknown>'}"
    raise PolarsQueryBuilderError(msg)


def _polars_string_function(
    expr: exp.Anonymous,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    func_name = (expr.name or "").lower()
    if len(expr.expressions) != _STRING_FUNC_ARG_COUNT:
        msg = f"{func_name} requires column and string literal arguments"
        raise PolarsQueryBuilderError(msg)
    return _polars_string_predicate_expr(
        expr.expressions[0],
        expr.expressions[1],
        func_name=func_name,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _polars_string_unary_expr(
    expr: exp.Expression | None,
    *,
    func_name: str,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    if func_name not in _STRING_UNARY_FUNCS:
        msg = f"Unsupported function: {func_name or '<unknown>'}"
        raise PolarsQueryBuilderError(msg)
    if expr is None or not isinstance(expr, exp.Column):
        msg = f"{func_name or 'function'} requires a column argument"
        raise PolarsQueryBuilderError(msg)
    column = _column_name(expr)
    _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="select")
    column_type = column_types.get(column) if column_types is not None else None
    if column_type is not None and column_type_base(column_type) != "VARCHAR":
        msg = f"{func_name} is only supported for VARCHAR columns"
        raise PolarsQueryBuilderError(msg)
    pl_mod = _require_polars()
    if func_name == "upper":
        return pl_mod.col(column).str.to_uppercase()
    return pl_mod.col(column).str.to_lowercase()


def _polars_string_predicate_expr(
    column_expr: exp.Expression | None,
    value_expr: exp.Expression | None,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
    func_name: str,
) -> PolarsExpr:
    if not isinstance(column_expr, exp.Column):
        msg = f"{func_name} requires a column argument"
        raise PolarsQueryBuilderError(msg)
    column = _column_name(column_expr)
    _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="select")
    column_type = column_types.get(column) if column_types is not None else None
    _validate_operator(op=_STRING_FUNC_MAP[func_name], column_type=column_type)
    value = _literal_value(value_expr)
    if not isinstance(value, str):
        msg = f"{func_name} requires a string literal"
        raise PolarsQueryBuilderError(msg)
    pl_mod = _require_polars()
    if func_name == "contains":
        return pl_mod.col(column).str.contains(value, literal=True)
    return pl_mod.col(column).str.starts_with(value)


def _polars_coalesce_expr(
    expressions: list[exp.Expression],
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    if not expressions:
        msg = "coalesce requires at least one argument"
        raise PolarsQueryBuilderError(msg)
    pl_mod = _require_polars()
    resolved = [
        _polars_expr_from_projection(
            item,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        for item in expressions
    ]
    coalesce = getattr(pl_mod, "coalesce", None)
    if callable(coalesce):
        return cast("PolarsExpr", coalesce(resolved))
    msg = "polars coalesce is unavailable"
    raise PolarsQueryBuilderError(msg)


def _where_expr_from_ast(
    *,
    ast: exp.Select,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr | None:
    where = ast.args.get("where")
    if where is None:
        return None
    predicate = where.this
    if predicate is None:
        return None
    return _build_predicate_expr(
        predicate,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _order_by_from_ast(
    ast: exp.Select,
    *,
    allowed_columns: frozenset[str],
) -> tuple[list[str], list[bool]]:
    order = ast.args.get("order")
    if order is None:
        return [], []
    columns: list[str] = []
    descending: list[bool] = []
    for item in order.expressions:
        if not isinstance(item, exp.Ordered):
            msg = f"Unsupported order_by expression: {type(item).__name__}"
            raise PolarsQueryBuilderError(msg)
        if not isinstance(item.this, exp.Column):
            msg = "Order by expressions must be columns"
            raise PolarsQueryBuilderError(msg)
        column = _column_name(item.this)
        _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="order_by")
        columns.append(column)
        descending.append(bool(item.args.get("desc")))
    return columns, descending


def _limit_offset_from_ast(ast: exp.Select) -> tuple[int | None, int]:
    limit_expr = ast.args.get("limit")
    offset_expr = ast.args.get("offset")
    limit_value: int | None = None
    offset_value = 0
    if limit_expr is not None:
        expression = limit_expr.expression
        if expression is None:
            expression = limit_expr.this
        limit_value = _literal_as_int(expression)
    if offset_expr is not None:
        expression = offset_expr.expression
        if expression is None:
            expression = offset_expr.this
        offset_value = _literal_as_int(expression)
    return limit_value, offset_value


def _literal_as_int(expr: exp.Expression | None) -> int:
    if expr is None:
        msg = "Expected literal for limit/offset"
        raise PolarsQueryBuilderError(msg)
    if isinstance(expr, exp.Literal):
        raw = expr.this
        if raw is None:
            msg = "Limit/offset literal is empty"
            raise PolarsQueryBuilderError(msg)
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            msg = "Limit/offset literal must be an integer"
            raise PolarsQueryBuilderError(msg) from exc
    msg = f"Unsupported limit/offset expression: {type(expr).__name__}"
    raise PolarsQueryBuilderError(msg)


def _build_predicate_expr(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    handler = _PREDICATE_DISPATCH.get(type(expr))
    if handler is not None:
        return handler(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.In):
        return _build_in_expr_ast(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.Contains):
        return _build_string_predicate_expr(
            func_name="contains",
            column_expr=expr.this,
            value_expr=expr.expression,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.StartsWith):
        return _build_string_predicate_expr(
            func_name="starts_with",
            column_expr=expr.this,
            value_expr=expr.expression,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.Anonymous):
        return _build_string_expr_ast(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, _AST_COMPARISON_TYPES):
        return _build_comparison_expr_ast(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    msg = f"Unsupported predicate expression: {type(expr).__name__}"
    raise PolarsQueryBuilderError(msg)


def _build_paren_predicate(
    expr: exp.Paren,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    if expr.this is None:
        msg = "Expected predicate inside parentheses"
        raise PolarsQueryBuilderError(msg)
    return _build_predicate_expr(
        expr.this,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _build_and_predicate(
    expr: exp.And,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    if expr.this is None or expr.expression is None:
        msg = "AND predicate requires two expressions"
        raise PolarsQueryBuilderError(msg)
    left = _build_predicate_expr(
        expr.this,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    right = _build_predicate_expr(
        expr.expression,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    return left & right


def _build_or_predicate(
    expr: exp.Or,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    if expr.this is None or expr.expression is None:
        msg = "OR predicate requires two expressions"
        raise PolarsQueryBuilderError(msg)
    left = _build_predicate_expr(
        expr.this,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    right = _build_predicate_expr(
        expr.expression,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    return left | right


def _build_not_predicate(
    expr: exp.Not,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    if expr.this is None:
        msg = "NOT predicate requires an expression"
        raise PolarsQueryBuilderError(msg)
    return ~_build_predicate_expr(
        expr.this,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


_PREDICATE_DISPATCH = {
    exp.Paren: _build_paren_predicate,
    exp.And: _build_and_predicate,
    exp.Or: _build_or_predicate,
    exp.Not: _build_not_predicate,
}


_AST_COMPARISON_TYPES: tuple[type[exp.Expression], ...] = (
    exp.EQ,
    exp.NEQ,
    exp.LT,
    exp.LTE,
    exp.GT,
    exp.GTE,
)
_AST_COMPARISON_OPS: dict[type[exp.Expression], str] = {
    exp.EQ: "eq",
    exp.NEQ: "ne",
    exp.LT: "lt",
    exp.LTE: "lte",
    exp.GT: "gt",
    exp.GTE: "gte",
}
_REVERSED_OPS: dict[str, str] = {
    "eq": "eq",
    "ne": "ne",
    "lt": "gt",
    "lte": "gte",
    "gt": "lt",
    "gte": "lte",
}
_PREDICATE_DISPATCH: dict[type[exp.Expression], Callable[..., PolarsExpr]]


def _build_comparison_expr_ast(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    op = _AST_COMPARISON_OPS[type(expr)]
    left = expr.this
    right = expr.expression
    if left is None or right is None:
        msg = "Comparison predicates require two expressions"
        raise PolarsQueryBuilderError(msg)
    if isinstance(left, exp.Column) and _is_literal(right):
        column = _column_name(left)
        value = _literal_value(right)
    elif isinstance(right, exp.Column) and _is_literal(left):
        column = _column_name(right)
        value = _literal_value(left)
        op = _REVERSED_OPS[op]
    else:
        msg = "Comparison predicates must compare a column to a literal"
        raise PolarsQueryBuilderError(msg)
    _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="filter")
    column_type = column_types.get(column) if column_types is not None else None
    _validate_operator(op=op, column_type=column_type)
    if isinstance(value, list):
        msg = f"{op} operator does not support list value"
        raise PolarsQueryBuilderError(msg)
    builder = _COMPARISON_BUILDERS[op]
    pl_mod = _require_polars()
    return builder(pl_mod.col(column), value)


def _build_in_expr_ast(
    expr: exp.In,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    if expr.this is None or not isinstance(expr.this, exp.Column):
        msg = "IN operator requires a column on the left"
        raise PolarsQueryBuilderError(msg)
    column = _column_name(expr.this)
    _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="filter")
    column_type = column_types.get(column) if column_types is not None else None
    _validate_operator(op="in", column_type=column_type)
    values = [_literal_value(item) for item in expr.expressions]
    pl_mod = _require_polars()
    if not values:
        return pl_mod.lit(value=False)
    return pl_mod.col(column).is_in(values)


def _build_string_expr_ast(
    expr: exp.Anonymous,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    func_name = (expr.name or "").lower()
    if func_name == "starts_with":
        op = "starts_with"
    elif func_name == "contains":
        op = "contains"
    else:
        msg = f"Unsupported function: {func_name or '<unknown>'}"
        raise PolarsQueryBuilderError(msg)
    if len(expr.expressions) != _STRING_FUNC_ARG_COUNT:
        msg = f"{op} requires column and string literal arguments"
        raise PolarsQueryBuilderError(msg)
    return _build_string_predicate_expr(
        func_name=op,
        column_expr=expr.expressions[0],
        value_expr=expr.expressions[1],
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _build_string_predicate_expr(
    *,
    func_name: str,
    column_expr: exp.Expression | None,
    value_expr: exp.Expression | None,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    if not isinstance(column_expr, exp.Column):
        msg = f"{func_name} requires a column argument"
        raise PolarsQueryBuilderError(msg)
    column = _column_name(column_expr)
    _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="filter")
    column_type = column_types.get(column) if column_types is not None else None
    op = "startswith" if func_name == "starts_with" else func_name
    _validate_operator(op=op, column_type=column_type)
    value = _literal_value(value_expr)
    if not isinstance(value, str):
        msg = f"{func_name} operator requires string value"
        raise PolarsQueryBuilderError(msg)
    pl_mod = _require_polars()
    if func_name == "contains":
        return pl_mod.col(column).str.contains(value, literal=True)
    return pl_mod.col(column).str.starts_with(value)


def _validate_operator(*, op: str, column_type: ColumnType | None) -> None:
    allowed_ops = allowed_ops_for_column_type(column_type)
    if op not in allowed_ops:
        msg = (
            f"Operator {op} is not supported for column type {column_type or _UNKNOWN_COLUMN_TYPE}"
        )
        raise PolarsQueryBuilderError(msg)
    base = column_type_base(column_type) if column_type is not None else None
    if op in _ORDERING_OPS and base == "VARCHAR":
        msg = f"Operator {op} is not supported for string columns"
        raise PolarsQueryBuilderError(msg)
    if op == "in" and base in {"JSON", "STRUCT", "MAP", "LIST", "UNION"}:
        msg = "IN operator is not supported for JSON columns"
        raise PolarsQueryBuilderError(msg)
    if op in _STRING_OPS and base is not None and base != "VARCHAR":
        msg = f"{op} operator is only supported for VARCHAR columns"
        raise PolarsQueryBuilderError(msg)


def _is_literal(expr: exp.Expression | None) -> bool:
    return isinstance(expr, (exp.Literal, exp.Boolean))


def _literal_value(expr: exp.Expression | None) -> FilterScalar:
    if expr is None:
        msg = "Expected literal value"
        raise PolarsQueryBuilderError(msg)
    value = _literal_from_to_py(expr)
    if value is not None:
        return value
    value = _literal_from_boolean(expr)
    if value is not None:
        return value
    value = _literal_from_literal(expr)
    if value is not None:
        return value
    msg = f"Unsupported literal type: {type(expr).__name__}"
    raise PolarsQueryBuilderError(msg)


def _literal_from_to_py(expr: exp.Expression) -> FilterScalar | None:
    to_py = getattr(expr, "to_py", None)
    if callable(to_py):
        try:
            value = to_py()
        except (TypeError, ValueError):
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
    return None


def _literal_from_boolean(expr: exp.Expression) -> FilterScalar | None:
    if not isinstance(expr, exp.Boolean):
        return None
    raw = expr.this
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
    return None


def _literal_from_literal(expr: exp.Expression) -> FilterScalar | None:
    result: FilterScalar | None = None
    if isinstance(expr, exp.Literal):
        raw = expr.this
        if getattr(expr, "is_string", False):
            result = str(raw)
        elif getattr(expr, "is_int", False):
            result = int(raw)
        elif getattr(expr, "is_number", False):
            result = float(raw)
        elif isinstance(raw, (int, float)):
            result = raw
        elif isinstance(raw, str):
            parsed = _parse_numeric_literal(raw)
            if parsed is None:
                msg = "Numeric literal could not be parsed"
                raise PolarsQueryBuilderError(msg)
            result = parsed
    return result


def _parse_numeric_literal(raw: str) -> int | float | None:
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return None


def _column_name(column: exp.Column) -> str:
    identifier = column.this
    if isinstance(identifier, exp.Identifier):
        name = identifier.this
    else:
        name = getattr(column, "name", None)
        if name is None:
            name = str(identifier)
    if not isinstance(name, str) or not name:
        msg = "Column name is missing"
        raise PolarsQueryBuilderError(msg)
    return name


_UNKNOWN_COLUMN_TYPE = "UNKNOWN"
_STRING_FUNC_ARG_COUNT = 2
_STRING_FUNC_MAP = {
    "contains": "contains",
    "starts_with": "startswith",
}
_FILTER_STRING_FUNCS = frozenset(_STRING_FUNC_MAP.keys())
_STRING_PREDICATE_FUNCS = frozenset({"contains", "starts_with"})
_STRING_UNARY_FUNCS = frozenset({"lower", "upper"})
_CASE_FUNCTIONS = frozenset({"coalesce"})
_COMPARISON_BUILDERS: dict[str, Callable[[PolarsExpr, FilterScalar], PolarsExpr]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "lte": operator.le,
    "gt": operator.gt,
    "gte": operator.ge,
}
_ORDERING_OPS = frozenset({"lt", "lte", "gt", "gte"})
_STRING_OPS = frozenset({"contains", "startswith"})


__all__ = [
    "PolarsQueryBuilderError",
    "apply_query_ast",
    "can_apply_query_ast",
]
