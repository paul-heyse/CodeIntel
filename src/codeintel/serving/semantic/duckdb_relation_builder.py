"""DuckDB relation-based query builder for semantic specs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb
import pyarrow as pa
from sqlglot import exp

from codeintel.core.columnar.schema_alignment import (
    align_reader_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.iceberg.guardrails import iceberg_enforced_table, require_iceberg_read
from codeintel.core.schemas.primitives import column_type_base, normalize_column_type
from codeintel.serving.semantic.filter_ops import allowed_ops_for_column_type
from codeintel.serving.semantic.iceberg_scans import (
    IcebergScanError,
    IcebergScanRequest,
    iceberg_scan_for_query,
    iceberg_table_exists,
)
from codeintel.serving.semantic.models import FilterSpec, FilterValue
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.storage.duckdb_types import (
    ColumnExpression,
    ConstantExpression,
    DuckDBCatalogException,
    DuckDBConnection,
    DuckDBRelation,
    Expression,
    FunctionExpression,
)
from codeintel.storage.helpers.json import normalize_duckdb_json_value

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from duckdb.typing import DuckDBPyType

    from codeintel.core.config.settings import IcebergSettings
    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.db.pointer import ServingSnapshotPointer
    from codeintel.serving.semantic.models import FilterScalar


class DuckDBRelationQueryBuilderError(ValueError):
    """Raised when a relation-based query cannot be built."""


DEFAULT_FRAGMENT_READAHEAD = 2


@dataclass(frozen=True, slots=True)
class RelationScanOptions:
    """Scan options for Arrow-backed DuckDB relations."""

    batch_size: int
    fragment_readahead: int | None = DEFAULT_FRAGMENT_READAHEAD
    metrics_enabled: bool = False


@dataclass(frozen=True, slots=True)
class RelationBuildContext:
    """Context required to build DuckDB relation plans."""

    scan_options: RelationScanOptions
    column_types: Mapping[str, ColumnType] | None = None
    contract_schema: pa.Schema | None = None
    iceberg: IcebergRelationContext | None = None


@dataclass(frozen=True, slots=True)
class IcebergRelationContext:
    """Inputs needed to resolve Iceberg scans inside DuckDB."""

    settings: IcebergSettings
    pointer: ServingSnapshotPointer


@dataclass(frozen=True, slots=True)
class IcebergScanInputs:
    """Iceberg scan inputs for query pushdown."""

    columns: Sequence[str]
    filters: Sequence[FilterSpec]
    order_by: Sequence[str]
    column_types: Mapping[str, ColumnType] | None


@dataclass(frozen=True, slots=True)
class _AstQueryComponents:
    select_exprs: list[Expression]
    predicate: Expression | None
    order_by: list[tuple[str, bool]]
    limit: int | None
    offset: int


@dataclass(frozen=True, slots=True)
class _PredicateContext:
    allowed_columns: frozenset[str]
    column_types: Mapping[str, ColumnType] | None
    ctx: str


def build_relation_plan(
    *,
    con: DuckDBConnection,
    spec: SemanticQuerySpec,
    ast: exp.Select,
    context: RelationBuildContext,
) -> DuckDBRelation:
    """Build a DuckDB relation plan for a semantic query spec.

    Returns
    -------
    DuckDBRelation
        Lazy relation representing the query plan.
    """
    iceberg_inputs = None
    if not _ast_has_joins(ast):
        iceberg_inputs = IcebergScanInputs(
            columns=spec.columns,
            filters=spec.filters,
            order_by=spec.order_by,
            column_types=context.column_types,
        )
    if _ast_has_joins(ast):
        relation = _relation_from_ast(
            con=con,
            ast=ast,
            context=context,
        )
    else:
        relation = _resolve_relation(
            con=con,
            table_key=spec.table_key,
            context=context,
            iceberg_inputs=iceberg_inputs,
        )
    return apply_query_ast(
        relation,
        ast=ast,
        allowed_columns=spec.allowed_columns,
        column_types=context.column_types,
    )


def apply_query_ast(
    relation: DuckDBRelation,
    *,
    ast: exp.Expression,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None = None,
) -> DuckDBRelation:
    """Apply a SQLGlot AST to a DuckDB relation.

    Returns
    -------
    DuckDBRelation
        Updated relation reflecting filters, ordering, and limits from the AST.

    Raises
    ------
    DuckDBRelationQueryBuilderError
        When the AST is invalid, uses unsupported constructs, or specifies an
        OFFSET without a LIMIT.
    """
    components = _parse_ast_components(
        ast=ast,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    if components.predicate is not None:
        relation = relation.filter(components.predicate)
    if components.order_by:
        relation = relation.order(_order_by_from_components(components.order_by))
    relation = relation.select(*components.select_exprs)
    if components.limit is not None or components.offset:
        if components.limit is None:
            msg = "OFFSET requires a LIMIT for DuckDB relation plans"
            raise DuckDBRelationQueryBuilderError(msg)
        relation = relation.limit(components.limit, offset=components.offset)
    return relation


def _resolve_relation(
    *,
    con: DuckDBConnection,
    table_key: str,
    context: RelationBuildContext,
    iceberg_inputs: IcebergScanInputs | None,
) -> DuckDBRelation:
    iceberg_ctx = context.iceberg
    enforced = False
    if iceberg_ctx is not None:
        enforced = iceberg_enforced_table(
            settings=iceberg_ctx.settings,
            table_key=table_key,
        )
    if iceberg_ctx is not None and enforced:
        require_iceberg_read(settings=iceberg_ctx.settings, table_key=table_key)
        if not iceberg_table_exists(settings=iceberg_ctx.settings, table_key=table_key):
            msg = f"Iceberg table missing for enforced table: {table_key}"
            raise DuckDBRelationQueryBuilderError(msg)
    if (
        iceberg_ctx is not None
        and iceberg_ctx.settings.read_enabled
        and iceberg_table_exists(settings=iceberg_ctx.settings, table_key=table_key)
    ):
        try:
            return _scan_iceberg(
                con=con,
                table_key=table_key,
                context=context,
                iceberg_inputs=iceberg_inputs,
            )
        except IcebergScanError as exc:
            if enforced:
                msg = f"Iceberg scan failed for enforced table: {table_key}"
                raise DuckDBRelationQueryBuilderError(msg) from exc
            if not iceberg_ctx.settings.read_fallback_enabled:
                msg = f"Iceberg scan failed with fallback disabled for {table_key}"
                raise DuckDBRelationQueryBuilderError(msg) from exc
            LOG.warning("Falling back to dataset scan for %s", table_key)
    try:
        return con.table(table_key)
    except DuckDBCatalogException as exc:
        msg = f"Unknown DuckDB table/view: {table_key}"
        raise DuckDBRelationQueryBuilderError(msg) from exc


def _ast_has_joins(ast: exp.Select) -> bool:
    joins = ast.args.get("joins")
    return bool(joins)


def _relation_from_ast(
    *,
    con: DuckDBConnection,
    ast: exp.Select,
    context: RelationBuildContext,
) -> DuckDBRelation:
    from_expr = ast.args.get("from_")
    if not isinstance(from_expr, exp.From):
        msg = "JOIN queries require a FROM clause"
        raise DuckDBRelationQueryBuilderError(msg)
    base_source = from_expr.this
    if not isinstance(base_source, exp.Table):
        msg = "JOIN base source must be a table"
        raise DuckDBRelationQueryBuilderError(msg)
    base_relation = _relation_for_table(
        con=con,
        table=base_source,
        context=context,
    )
    relation = _apply_relation_alias(base_relation, base_source)
    joins = ast.args.get("joins") or []
    for join in joins:
        if not isinstance(join, exp.Join):
            msg = f"Unsupported join expression: {type(join).__name__}"
            raise DuckDBRelationQueryBuilderError(msg)
        relation = _apply_join(
            relation=relation,
            con=con,
            join=join,
            context=context,
        )
    return relation


def _relation_for_table(
    *,
    con: DuckDBConnection,
    table: exp.Table,
    context: RelationBuildContext,
) -> DuckDBRelation:
    table_key = _table_key_from_table(table)
    return _resolve_relation(
        con=con,
        table_key=table_key,
        context=context,
        iceberg_inputs=None,
    )


def _apply_relation_alias(relation: DuckDBRelation, table: exp.Table) -> DuckDBRelation:
    alias = table.alias_or_name
    if not alias:
        return relation
    setter = getattr(relation, "set_alias", None)
    if not callable(setter):
        return relation
    try:
        aliased = setter(str(alias))
    except TypeError:
        return relation
    return aliased if isinstance(aliased, duckdb.DuckDBPyRelation) else relation


def _apply_join(
    *,
    relation: DuckDBRelation,
    con: DuckDBConnection,
    join: exp.Join,
    context: RelationBuildContext,
) -> DuckDBRelation:
    if not isinstance(join.this, exp.Table):
        msg = "JOIN targets must be tables"
        raise DuckDBRelationQueryBuilderError(msg)
    join_relation = _relation_for_table(con=con, table=join.this, context=context)
    join_relation = _apply_relation_alias(join_relation, join.this)
    join_condition = _join_condition_expr(join.args.get("on"))
    join_type = _join_type(join)
    try:
        return relation.join(join_relation, join_condition, how=join_type)
    except (TypeError, ValueError) as exc:
        msg = f"Failed to apply JOIN ({join_type})"
        raise DuckDBRelationQueryBuilderError(msg) from exc


def _join_type(join: exp.Join) -> str:
    side = join.args.get("side")
    if not isinstance(side, str):
        return "inner"
    normalized = side.strip().lower()
    if normalized == "full":
        return "outer"
    return normalized


def _join_condition_expr(expr: exp.Expression | None) -> Expression:
    if expr is None:
        msg = "JOIN requires an ON clause"
        raise DuckDBRelationQueryBuilderError(msg)
    if isinstance(expr, exp.Paren):
        return _join_condition_expr(expr.this)
    if isinstance(expr, exp.And):
        if expr.this is None or expr.expression is None:
            msg = "JOIN AND requires two expressions"
            raise DuckDBRelationQueryBuilderError(msg)
        return _join_condition_expr(expr.this) & _join_condition_expr(expr.expression)
    if isinstance(expr, exp.EQ):
        left = expr.this
        right = expr.expression
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            msg = "JOIN conditions must compare columns"
            raise DuckDBRelationQueryBuilderError(msg)
        left_expr = _qualified_column(left)
        right_expr = _qualified_column(right)
        return left_expr == right_expr
    msg = f"Unsupported JOIN predicate: {type(expr).__name__}"
    raise DuckDBRelationQueryBuilderError(msg)


def _qualified_column(column: exp.Column) -> Expression:
    name = _column_name(column)
    table = column.table
    if isinstance(table, exp.Identifier):
        qualifier = table.this
    elif isinstance(table, str):
        qualifier = table
    else:
        qualifier = None
    column_name = f"{qualifier}.{name}" if qualifier else name
    return ColumnExpression(column_name)


def _table_key_from_table(table: exp.Table) -> str:
    name = table.name
    schema = table.db
    if schema:
        return f"{schema}.{name}"
    return name


def _scan_iceberg(
    *,
    con: DuckDBConnection,
    table_key: str,
    context: RelationBuildContext,
    iceberg_inputs: IcebergScanInputs | None,
) -> DuckDBRelation:
    iceberg_ctx = context.iceberg
    if iceberg_ctx is None:
        msg = "Iceberg context is required for Iceberg scans"
        raise IcebergScanError(msg)
    inputs = iceberg_inputs or IcebergScanInputs(
        columns=(),
        filters=(),
        order_by=(),
        column_types=None,
    )
    scan_result = iceberg_scan_for_query(
        request=IcebergScanRequest(
            table_key=table_key,
            columns=inputs.columns,
            filters=list(inputs.filters),
            order_by=inputs.order_by,
            column_types=inputs.column_types,
            pointer=iceberg_ctx.pointer,
            settings=iceberg_ctx.settings,
            batch_size=context.scan_options.batch_size,
        )
    )
    reader = scan_result.scan.to_arrow_batch_reader()
    if context.contract_schema is not None:
        aligned = align_reader_to_contract(
            reader,
            context.contract_schema,
            extras_policy=extras_policy_from_schema(context.contract_schema),
        )
        try:
            return con.from_arrow(aligned)
        except (duckdb.Error, TypeError, ValueError):
            reader = scan_result.scan.to_arrow_batch_reader()
    try:
        return con.from_arrow(reader)
    except (duckdb.Error, TypeError, ValueError) as exc:
        msg = f"Iceberg scan relation build failed for {table_key}"
        raise DuckDBRelationQueryBuilderError(msg) from exc


def _validate_pagination(*, limit: int, offset: int) -> None:
    if limit < 0:
        msg = "limit must be >= 0"
        raise DuckDBRelationQueryBuilderError(msg)
    if offset < 0:
        msg = "offset must be >= 0"
        raise DuckDBRelationQueryBuilderError(msg)


def _require_allowed_column(*, column: str, allowed_columns: frozenset[str], ctx: str) -> None:
    if column not in allowed_columns:
        msg = f"Unknown {ctx} column: {column}"
        raise DuckDBRelationQueryBuilderError(msg)


def _parse_ast_components(
    *,
    ast: exp.Expression,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> _AstQueryComponents:
    if not isinstance(ast, exp.Select):
        msg = "Expected SQLGlot Select expression"
        raise DuckDBRelationQueryBuilderError(msg)
    select_exprs = _select_exprs_from_ast(
        ast,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    predicate = _where_predicate_from_ast(
        ast=ast,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    order_by = _order_by_from_ast(ast, allowed_columns=allowed_columns)
    limit, offset = _limit_offset_from_ast(ast)
    _validate_pagination(limit=limit or 0, offset=offset)
    return _AstQueryComponents(
        select_exprs=select_exprs,
        predicate=predicate,
        order_by=order_by,
        limit=limit,
        offset=offset,
    )


def _select_exprs_from_ast(
    ast: exp.Select,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> list[Expression]:
    select_exprs = [
        _duckdb_expr_from_select(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        for expr in ast.expressions
    ]
    if not select_exprs:
        msg = "Select expression must include at least one column"
        raise DuckDBRelationQueryBuilderError(msg)
    return select_exprs


def _duckdb_expr_from_select(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if isinstance(expr, exp.Alias):
        alias = expr.alias
        if not alias:
            msg = "Select alias requires a name"
            raise DuckDBRelationQueryBuilderError(msg)
        base = _duckdb_expr_from_select(
            expr.this,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        return base.alias(alias)
    return _duckdb_expr_from_projection(
        expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _duckdb_expr_from_projection(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    result: Expression | None = None
    predicate_context = _PredicateContext(
        allowed_columns=allowed_columns,
        column_types=column_types,
        ctx="select",
    )
    if isinstance(expr, exp.Column):
        column = _column_name(expr)
        _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="select")
        result = ColumnExpression(column)
    elif isinstance(expr, exp.Lower):
        result = _duckdb_string_unary_expr(
            expr.this,
            func_name="lower",
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, exp.Upper):
        result = _duckdb_string_unary_expr(
            expr.this,
            func_name="upper",
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, exp.Coalesce):
        result = _duckdb_coalesce_expr(
            expr.expressions,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, exp.Contains):
        result = _duckdb_string_predicate_expr(
            expr.this,
            expr.expression,
            func_name="contains",
            context=predicate_context,
        )
    elif isinstance(expr, exp.StartsWith):
        result = _duckdb_string_predicate_expr(
            expr.this,
            expr.expression,
            func_name="starts_with",
            context=predicate_context,
        )
    elif isinstance(expr, exp.Anonymous):
        result = _duckdb_function_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, (exp.Boolean, exp.Literal)):
        result = ConstantExpression(_literal_value(expr))

    if result is None:
        msg = f"Unsupported select expression: {type(expr).__name__}"
        raise DuckDBRelationQueryBuilderError(msg)
    return result


def _duckdb_function_expr(
    expr: exp.Anonymous,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    func_name = (expr.name or "").lower()
    if func_name in _STRING_PREDICATE_FUNCS:
        return _duckdb_string_function(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if func_name in _STRING_UNARY_FUNCS:
        return _duckdb_string_unary_function(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if func_name == "coalesce":
        return _duckdb_coalesce_function(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    msg = f"Unsupported function: {func_name or '<unknown>'}"
    raise DuckDBRelationQueryBuilderError(msg)


def _duckdb_string_function(
    expr: exp.Anonymous,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    func_name = (expr.name or "").lower()
    if len(expr.expressions) != _STRING_FUNC_ARG_COUNT:
        msg = f"{func_name} requires column and string literal arguments"
        raise DuckDBRelationQueryBuilderError(msg)
    context = _PredicateContext(
        allowed_columns=allowed_columns,
        column_types=column_types,
        ctx="select",
    )
    return _duckdb_string_predicate_expr(
        expr.expressions[0],
        expr.expressions[1],
        func_name=func_name,
        context=context,
    )


def _duckdb_string_unary_function(
    expr: exp.Anonymous,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    func_name = (expr.name or "").lower()
    if len(expr.expressions) != 1:
        msg = f"{func_name} requires a single column argument"
        raise DuckDBRelationQueryBuilderError(msg)
    return _duckdb_string_unary_expr(
        expr.expressions[0],
        func_name=func_name,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _duckdb_coalesce_function(
    expr: exp.Anonymous,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    return _duckdb_coalesce_expr(
        expr.expressions,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _duckdb_string_predicate_expr(
    column_expr: exp.Expression | None,
    value_expr: exp.Expression | None,
    *,
    func_name: str,
    context: _PredicateContext,
) -> Expression:
    if not isinstance(column_expr, exp.Column):
        msg = f"{func_name} requires a column argument"
        raise DuckDBRelationQueryBuilderError(msg)
    column = _column_name(column_expr)
    _require_allowed_column(
        column=column,
        allowed_columns=context.allowed_columns,
        ctx=context.ctx,
    )
    column_type = context.column_types.get(column) if context.column_types is not None else None
    op = _STRING_FUNC_MAP[func_name]
    _validate_operator(op=op, column_type=column_type)
    value = _literal_value(value_expr)
    if not isinstance(value, str):
        msg = f"{func_name} requires a string literal"
        raise DuckDBRelationQueryBuilderError(msg)
    return FunctionExpression(func_name, ColumnExpression(column), ConstantExpression(value))


def _duckdb_string_unary_expr(
    column_expr: exp.Expression | None,
    *,
    func_name: str,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if func_name not in _STRING_UNARY_FUNCS:
        msg = f"Unsupported function: {func_name or '<unknown>'}"
        raise DuckDBRelationQueryBuilderError(msg)
    if not isinstance(column_expr, exp.Column):
        msg = f"{func_name} requires a column argument"
        raise DuckDBRelationQueryBuilderError(msg)
    column = _column_name(column_expr)
    _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="select")
    column_type = column_types.get(column) if column_types is not None else None
    if column_type is not None and column_type_base(column_type) != "VARCHAR":
        msg = f"{func_name} is only supported for VARCHAR columns"
        raise DuckDBRelationQueryBuilderError(msg)
    return FunctionExpression(func_name, ColumnExpression(column))


def _duckdb_coalesce_expr(
    expressions: list[exp.Expression],
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if not expressions:
        msg = "coalesce requires at least one argument"
        raise DuckDBRelationQueryBuilderError(msg)
    args = [
        _duckdb_expr_from_projection(
            item,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        for item in expressions
    ]
    return FunctionExpression("coalesce", *args)


def _where_predicate_from_ast(
    *,
    ast: exp.Select,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression | None:
    where = ast.args.get("where")
    if where is None:
        return None
    predicate = where.this
    if predicate is None:
        return None
    return _build_predicate_expr_ast(
        predicate,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _order_by_from_ast(
    ast: exp.Select,
    *,
    allowed_columns: frozenset[str],
) -> list[tuple[str, bool]]:
    order = ast.args.get("order")
    if order is None:
        return []
    items: list[tuple[str, bool]] = []
    for expr in order.expressions:
        if not isinstance(expr, exp.Ordered):
            msg = f"Unsupported order_by expression: {type(expr).__name__}"
            raise DuckDBRelationQueryBuilderError(msg)
        if not isinstance(expr.this, exp.Column):
            msg = "Order by expressions must be columns"
            raise DuckDBRelationQueryBuilderError(msg)
        column = _column_name(expr.this)
        _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="order_by")
        items.append((column, bool(expr.args.get("desc"))))
    return items


def _order_by_from_components(order_by: Sequence[tuple[str, bool]]) -> str:
    parts: list[str] = []
    for column, desc in order_by:
        suffix = " DESC" if desc else ""
        parts.append(f"{column}{suffix}")
    return ", ".join(parts)


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
        raise DuckDBRelationQueryBuilderError(msg)
    if isinstance(expr, exp.Literal):
        raw = expr.this
        if raw is None:
            msg = "Limit/offset literal is empty"
            raise DuckDBRelationQueryBuilderError(msg)
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            msg = "Limit/offset literal must be an integer"
            raise DuckDBRelationQueryBuilderError(msg) from exc
    msg = f"Unsupported limit/offset expression: {type(expr).__name__}"
    raise DuckDBRelationQueryBuilderError(msg)


def _build_predicate_expr_ast(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    handler = _AST_PREDICATE_DISPATCH.get(type(expr))
    if handler is not None:
        return handler(expr, allowed_columns=allowed_columns, column_types=column_types)
    if isinstance(expr, exp.In):
        return _build_in_expr_ast(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    predicate_context = _PredicateContext(
        allowed_columns=allowed_columns,
        column_types=column_types,
        ctx="filter",
    )
    if isinstance(expr, exp.Contains):
        return _duckdb_string_predicate_expr(
            expr.this,
            expr.expression,
            func_name="contains",
            context=predicate_context,
        )
    if isinstance(expr, exp.StartsWith):
        return _duckdb_string_predicate_expr(
            expr.this,
            expr.expression,
            func_name="starts_with",
            context=predicate_context,
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
    raise DuckDBRelationQueryBuilderError(msg)


def _build_paren_predicate_ast(
    expr: exp.Paren,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if expr.this is None:
        msg = "Expected predicate inside parentheses"
        raise DuckDBRelationQueryBuilderError(msg)
    return _build_predicate_expr_ast(
        expr.this,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _build_and_predicate_ast(
    expr: exp.And,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if expr.this is None or expr.expression is None:
        msg = "AND predicate requires two expressions"
        raise DuckDBRelationQueryBuilderError(msg)
    left = _build_predicate_expr_ast(
        expr.this,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    right = _build_predicate_expr_ast(
        expr.expression,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    return left & right


def _build_or_predicate_ast(
    expr: exp.Or,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if expr.this is None or expr.expression is None:
        msg = "OR predicate requires two expressions"
        raise DuckDBRelationQueryBuilderError(msg)
    left = _build_predicate_expr_ast(
        expr.this,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    right = _build_predicate_expr_ast(
        expr.expression,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    return left | right


def _build_not_predicate_ast(
    expr: exp.Not,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if expr.this is None:
        msg = "NOT predicate requires an expression"
        raise DuckDBRelationQueryBuilderError(msg)
    return ~_build_predicate_expr_ast(
        expr.this,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


_AST_PREDICATE_DISPATCH: dict[type[exp.Expression], Callable[..., Expression]] = {
    exp.Paren: _build_paren_predicate_ast,
    exp.And: _build_and_predicate_ast,
    exp.Or: _build_or_predicate_ast,
    exp.Not: _build_not_predicate_ast,
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


def _build_comparison_expr_ast(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    op = _AST_COMPARISON_OPS[type(expr)]
    left = expr.this
    right = expr.expression
    if left is None or right is None:
        msg = "Comparison predicates require two expressions"
        raise DuckDBRelationQueryBuilderError(msg)
    if isinstance(left, exp.Column) and _is_literal(right):
        column = _column_name(left)
        value = _literal_value(right)
    elif isinstance(right, exp.Column) and _is_literal(left):
        column = _column_name(right)
        value = _literal_value(left)
        op = _REVERSED_OPS[op]
    else:
        msg = "Comparison predicates must compare a column to a literal"
        raise DuckDBRelationQueryBuilderError(msg)
    _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="filter")
    column_type = column_types.get(column) if column_types is not None else None
    _validate_operator(op=op, column_type=column_type)
    return _build_comparison_predicate(
        col_expr=ColumnExpression(column),
        op=op,
        value=value,
        column_type=column_type,
    )


def _build_in_expr_ast(
    expr: exp.In,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if expr.this is None or not isinstance(expr.this, exp.Column):
        msg = "IN operator requires a column on the left"
        raise DuckDBRelationQueryBuilderError(msg)
    column = _column_name(expr.this)
    _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="filter")
    column_type = column_types.get(column) if column_types is not None else None
    _validate_operator(op="in", column_type=column_type)
    values = [_literal_value(item) for item in expr.expressions]
    return _build_in_predicate(
        col_expr=ColumnExpression(column),
        value=values,
        column_type=column_type,
    )


def _build_string_expr_ast(
    expr: exp.Anonymous,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    func_name = (expr.name or "").lower()
    if func_name == "starts_with":
        op = "startswith"
    elif func_name == "contains":
        op = "contains"
    else:
        msg = f"Unsupported function: {func_name or '<unknown>'}"
        raise DuckDBRelationQueryBuilderError(msg)
    if len(expr.expressions) != _STRING_FUNC_ARG_COUNT:
        msg = f"{op} requires column and string literal arguments"
        raise DuckDBRelationQueryBuilderError(msg)
    column_expr = expr.expressions[0]
    value_expr = expr.expressions[1]
    if not isinstance(column_expr, exp.Column):
        msg = f"{op} requires a column argument"
        raise DuckDBRelationQueryBuilderError(msg)
    column = _column_name(column_expr)
    _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="filter")
    column_type = column_types.get(column) if column_types is not None else None
    _validate_operator(op=op, column_type=column_type)
    value = _literal_value(value_expr)
    if not isinstance(value, str):
        msg = f"{op} operator requires string value"
        raise DuckDBRelationQueryBuilderError(msg)
    return _build_string_predicate(
        col_expr=ColumnExpression(column),
        op=op,
        value=value,
        column_type=column_type,
    )


def _validate_operator(*, op: str, column_type: ColumnType | None) -> None:
    allowed_ops = allowed_ops_for_column_type(column_type)
    if op not in allowed_ops:
        msg = (
            f"Operator {op} is not supported for column type {column_type or _UNKNOWN_COLUMN_TYPE}"
        )
        raise DuckDBRelationQueryBuilderError(msg)
    base = column_type_base(column_type) if column_type is not None else None
    if op in _ORDERING_OPS and base == "VARCHAR":
        msg = f"Operator {op} is not supported for string columns"
        raise DuckDBRelationQueryBuilderError(msg)
    if op == "in" and base in {"JSON", "STRUCT", "MAP", "LIST", "UNION"}:
        msg = "IN operator is not supported for JSON columns"
        raise DuckDBRelationQueryBuilderError(msg)
    if op in _STRING_OPS and base is not None and base != "VARCHAR":
        msg = f"{op} operator is only supported for VARCHAR columns"
        raise DuckDBRelationQueryBuilderError(msg)


def _is_literal(expr: exp.Expression | None) -> bool:
    return isinstance(expr, (exp.Literal, exp.Boolean))


def _literal_value(expr: exp.Expression | None) -> FilterScalar:
    if expr is None:
        msg = "Expected literal value"
        raise DuckDBRelationQueryBuilderError(msg)
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
    raise DuckDBRelationQueryBuilderError(msg)


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
        else:
            try:
                result = int(raw)
            except (TypeError, ValueError):
                try:
                    result = float(raw)
                except (TypeError, ValueError):
                    result = None
    return result


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
        raise DuckDBRelationQueryBuilderError(msg)
    return name


_UNKNOWN_COLUMN_TYPE = "UNKNOWN"
_ORDERING_OPS = frozenset({"lt", "lte", "gt", "gte"})
_STRING_OPS = frozenset({"contains", "startswith"})
_STRING_FUNC_ARG_COUNT = 2
_STRING_FUNC_MAP = {
    "contains": "contains",
    "starts_with": "startswith",
}
_STRING_PREDICATE_FUNCS = frozenset(_STRING_FUNC_MAP.keys())
_STRING_UNARY_FUNCS = frozenset({"lower", "upper"})
_DECIMAL_38_0 = "DECIMAL(38,0)"


def _duckdb_type_for_column(column_type: ColumnType | None) -> DuckDBPyType | None:
    if column_type is None:
        return None
    normalized = normalize_column_type(str(column_type))
    base = column_type_base(normalized)
    if normalized.upper().replace(" ", "") == _DECIMAL_38_0:
        return duckdb.sqltype(_DECIMAL_38_0)
    if base == "DECIMAL":
        if normalized.upper().startswith("DECIMAL("):
            return duckdb.sqltype(normalized)
        return duckdb.sqltype("DECIMAL")
    if base in {"STRUCT", "MAP", "LIST", "UNION"}:
        return duckdb.sqltype(normalized)
    type_map: dict[str, DuckDBPyType] = {
        "BOOLEAN": duckdb.sqltype("BOOLEAN"),
        "INTEGER": duckdb.sqltype("INTEGER"),
        "BIGINT": duckdb.sqltype("BIGINT"),
        "DOUBLE": duckdb.sqltype("DOUBLE"),
        "VARCHAR": duckdb.sqltype("VARCHAR"),
        "TIMESTAMP": duckdb.sqltype("TIMESTAMP"),
        "TIMESTAMPTZ": duckdb.sqltype("TIMESTAMPTZ"),
        "JSON": duckdb.sqltype("JSON"),
    }
    return type_map.get(base)


def _typed_constant(value: FilterValue, *, column_type: ColumnType | None) -> Expression:
    literal_value: object = value
    base = column_type_base(column_type) if column_type is not None else None
    if base == "JSON":
        literal_value = normalize_duckdb_json_value(value)
    literal = ConstantExpression(literal_value)
    if column_type is None:
        return literal
    if base in {"BOOLEAN", "INTEGER", "BIGINT", "DOUBLE", "VARCHAR"}:
        return literal
    duckdb_type = _duckdb_type_for_column(column_type)
    if duckdb_type is None:
        return literal
    return literal.cast(duckdb_type)


def _build_comparison_predicate(
    *,
    col_expr: Expression,
    op: str,
    value: FilterValue,
    column_type: ColumnType | None,
) -> Expression:
    if isinstance(value, list):
        msg = f"{op} operator does not support list value"
        raise DuckDBRelationQueryBuilderError(msg)
    base = column_type_base(column_type) if column_type is not None else None
    if op in _ORDERING_OPS and base == "VARCHAR":
        msg = f"Operator {op} is not supported for string columns"
        raise DuckDBRelationQueryBuilderError(msg)
    literal = _typed_constant(value, column_type=column_type)
    if op == "eq":
        return col_expr == literal
    if op == "ne":
        return col_expr != literal
    if op == "lt":
        return col_expr < literal
    if op == "lte":
        return col_expr <= literal
    if op == "gt":
        return col_expr > literal
    if op == "gte":
        return col_expr >= literal
    msg = f"Unsupported comparison operator: {op}"
    raise DuckDBRelationQueryBuilderError(msg)


def _build_in_predicate(
    *,
    col_expr: Expression,
    value: FilterValue,
    column_type: ColumnType | None,
) -> Expression:
    if not isinstance(value, list):
        msg = "IN operator requires list value"
        raise DuckDBRelationQueryBuilderError(msg)
    base = column_type_base(column_type) if column_type is not None else None
    if base in {"JSON", "STRUCT", "MAP", "LIST", "UNION"}:
        msg = "IN operator is not supported for JSON columns"
        raise DuckDBRelationQueryBuilderError(msg)
    if not value:
        return ConstantExpression(0) == ConstantExpression(1)
    constants = [_typed_constant(item, column_type=column_type) for item in value]
    return col_expr.isin(*constants)


def _build_string_predicate(
    *,
    col_expr: Expression,
    op: str,
    value: FilterValue,
    column_type: ColumnType | None,
) -> Expression:
    if not isinstance(value, str):
        msg = f"{op} operator requires string value"
        raise DuckDBRelationQueryBuilderError(msg)
    base = column_type_base(column_type) if column_type is not None else None
    if base is not None and base != "VARCHAR":
        msg = f"{op} operator is only supported for VARCHAR columns"
        raise DuckDBRelationQueryBuilderError(msg)
    literal = _typed_constant(value, column_type=column_type)
    func_name = "contains" if op == "contains" else "starts_with"
    return FunctionExpression(func_name, col_expr, literal)


def _order_by_expr(order_by: list[str], *, allowed_columns: frozenset[str]) -> str:
    order_parts: list[str] = []
    for col in order_by:
        descending = col.startswith("-")
        col_name = col[1:] if descending else col
        _require_allowed_column(column=col_name, allowed_columns=allowed_columns, ctx="order_by")
        suffix = " DESC" if descending else ""
        order_parts.append(f"{col_name}{suffix}")
    return ", ".join(order_parts)


__all__ = [
    "DuckDBRelationQueryBuilderError",
    "IcebergRelationContext",
    "RelationBuildContext",
    "RelationScanOptions",
    "apply_query_ast",
    "build_relation_plan",
]
