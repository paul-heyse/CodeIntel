"""DuckDB relation-based query builder for semantic specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
from sqlglot import exp

from codeintel.core.columnar.schema import DEFAULT_SCHEMA_PROMOTE_OPTIONS, SchemaPromoteOptions
from codeintel.core.columnar.schema_alignment import (
    align_reader_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.filters import FilterOpError, validate_filter_value
from codeintel.core.helpers.json import normalize_duckdb_json_value
from codeintel.core.schemas.primitives import column_type_base
from codeintel.core.schemas.type_mappings import complex_type_mapping
from codeintel.serving.semantic.duckdb_scan_adapter import scan_arrow, scan_parquet
from codeintel.serving.semantic.models import FilterValue, Op
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.storage.datasets.manifest_index import (
    DatasetScannerOptions,
    dataset_filter_expression,
    dataset_for_entry,
    dataset_scanner_for_entry,
    dataset_schema_for_entry,
)
from codeintel.storage.datasets.scanning import QueryPlanSpec
from codeintel.storage.duckdb_types import (
    ColumnExpression,
    ConstantExpression,
    DuckDBConnection,
    DuckDBRelation,
    Expression,
    FunctionExpression,
    duckdb_type_for_column_type,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from duckdb.typing import DuckDBPyType

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.semantic.models import FilterScalar
    from codeintel.storage.datasets.manifest_index import DatasetManifestEntry, DatasetManifestIndex


class DuckDBRelationQueryBuilderError(ValueError):
    """Raised when a relation-based query cannot be built."""


DEFAULT_FRAGMENT_READAHEAD = 2


@dataclass(frozen=True, slots=True)
class RelationScanOptions:
    """Scan options for Arrow-backed DuckDB relations."""

    batch_size: int
    batch_readahead: int | None = None
    fragment_readahead: int | None = DEFAULT_FRAGMENT_READAHEAD
    use_threads: bool | None = None
    unify_schemas: bool = False
    schema_promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS
    metrics_enabled: bool = False


@dataclass(frozen=True, slots=True)
class RelationBuildContext:
    """Context required to build DuckDB relation plans."""

    dataset_manifests: DatasetManifestIndex
    scan_options: RelationScanOptions
    column_types: Mapping[str, ColumnType] | None = None
    contract_schema: pa.Schema | None = None


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
    plan_spec: QueryPlanSpec | None = None,
) -> DuckDBRelation:
    """Build a DuckDB relation plan for a semantic query spec.

    Returns
    -------
    DuckDBRelation
        Lazy relation representing the query plan.
    """
    filter_expression = (
        plan_spec.filter_expression
        if plan_spec is not None
        else dataset_filter_expression(
            filters=spec.filters,
            allowed_columns=spec.allowed_columns,
            column_types=context.column_types,
        )
    )
    projection_columns = None
    if not _ast_has_joins(ast):
        ast_columns = _projection_columns_from_ast(ast)
        if ast_columns is None:
            projection_columns = None
        elif plan_spec is not None and plan_spec.columns:
            projection_columns = tuple(sorted(set(plan_spec.columns) | set(ast_columns)))
        else:
            projection_columns = ast_columns
    if _ast_has_joins(ast):
        relation = _relation_from_ast(
            con=con,
            ast=ast,
            context=context,
            allowed_columns=spec.allowed_columns,
        )
    else:
        relation = _resolve_relation(
            con=con,
            table_key=spec.table_key,
            context=context,
            filter_expression=filter_expression,
            projection_columns=projection_columns,
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


def validate_query_ast(
    *,
    ast: exp.Expression,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None = None,
) -> None:
    """Validate that a SQLGlot AST can be applied to a relation.

    Parameters
    ----------
    ast
        SQLGlot expression to validate.
    allowed_columns
        Allowed column set for selection/filtering/ordering.
    column_types
        Optional column type mapping for operator validation.
    """
    _parse_ast_components(
        ast=ast,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _resolve_relation(
    *,
    con: DuckDBConnection,
    table_key: str,
    context: RelationBuildContext,
    filter_expression: ds.Expression | None,
    projection_columns: Sequence[str] | None = None,
) -> DuckDBRelation:
    entry = context.dataset_manifests.get(table_key)
    if entry is None:
        msg = f"Unknown dataset entry: {table_key}"
        raise DuckDBRelationQueryBuilderError(msg)
    return _scan_dataset(
        con=con,
        entry=entry,
        context=context,
        filter_expression=filter_expression,
        projection_columns=projection_columns,
    )


def _ast_has_joins(ast: exp.Select) -> bool:
    joins = ast.args.get("joins")
    return bool(joins)


def _projection_columns_from_ast(ast: exp.Expression) -> tuple[str, ...] | None:
    if any(isinstance(node, exp.Star) for node in ast.find_all(exp.Star)):
        return None
    columns = {
        column.name for column in ast.find_all(exp.Column) if column.name and column.name != "*"
    }
    if not columns:
        return None
    return tuple(sorted(columns))


def _relation_from_ast(
    *,
    con: DuckDBConnection,
    ast: exp.Select,
    context: RelationBuildContext,
    allowed_columns: frozenset[str],
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
            allowed_columns=allowed_columns,
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
        filter_expression=None,
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


def _true_relation_expr() -> Expression:
    return ConstantExpression(1) == ConstantExpression(1)


def _apply_join(
    *,
    relation: DuckDBRelation,
    con: DuckDBConnection,
    join: exp.Join,
    context: RelationBuildContext,
    allowed_columns: frozenset[str],
) -> DuckDBRelation:
    if not isinstance(join.this, exp.Table):
        msg = "JOIN targets must be tables"
        raise DuckDBRelationQueryBuilderError(msg)
    if join.args.get("natural"):
        msg = "NATURAL JOIN is not supported in relation plans"
        raise DuckDBRelationQueryBuilderError(msg)
    join_relation = _relation_for_table(con=con, table=join.this, context=context)
    join_relation = _apply_relation_alias(join_relation, join.this)
    join_type = _join_type(join)
    join_condition = _join_condition_expr(
        join,
        join_type=join_type,
        allowed_columns=allowed_columns,
        column_types=context.column_types,
    )
    if join_type == "cross":
        cross_join = getattr(relation, "cross_join", None)
        if callable(cross_join):
            try:
                result = cross_join(join_relation)
            except (TypeError, ValueError) as exc:
                msg = "Failed to apply CROSS JOIN"
                raise DuckDBRelationQueryBuilderError(msg) from exc
            return (
                result
                if isinstance(result, duckdb.DuckDBPyRelation)
                else relation.join(join_relation, _true_relation_expr(), how="cross")
            )
        return relation.join(join_relation, _true_relation_expr(), how="cross")
    if join_condition is None:
        msg = "JOIN requires an ON or USING clause"
        raise DuckDBRelationQueryBuilderError(msg)
    try:
        return relation.join(join_relation, join_condition, how=join_type)
    except (TypeError, ValueError) as exc:
        msg = f"Failed to apply JOIN ({join_type})"
        raise DuckDBRelationQueryBuilderError(msg) from exc


_JOIN_TYPE_MAP: dict[str, str] = {
    "full": "outer",
    "full outer": "outer",
    "left outer": "left",
    "left": "left",
    "right outer": "right",
    "right": "right",
    "semi": "semi",
    "left semi": "semi",
    "leftsemi": "semi",
    "anti": "anti",
    "left anti": "anti",
    "leftanti": "anti",
    "cross": "cross",
    "cross join": "cross",
    "inner": "inner",
}


def _join_type(join: exp.Join) -> str:
    normalized = _normalize_join_type(join)
    if not normalized:
        return "inner"
    resolved = _JOIN_TYPE_MAP.get(normalized)
    if resolved is None:
        msg = f"Unsupported join type: {normalized}"
        raise DuckDBRelationQueryBuilderError(msg)
    return resolved


def _normalize_join_type(join: exp.Join) -> str:
    tokens: list[str] = []
    for value in (join.args.get("side"), join.args.get("kind"), join.args.get("method")):
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned:
                tokens.append(cleaned)
    normalized = " ".join(tokens).replace("_", " ").replace("-", " ").strip()
    return " ".join(normalized.split())


def _join_condition_expr(
    join: exp.Join,
    *,
    join_type: str,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression | None:
    using_expr = join.args.get("using")
    on_expr = join.args.get("on")
    if join_type == "cross":
        if using_expr is not None or on_expr is not None:
            msg = "CROSS JOIN cannot include ON or USING clauses"
            raise DuckDBRelationQueryBuilderError(msg)
        return None
    if using_expr is not None:
        right_alias = _join_table_alias(join.this)
        return _join_condition_from_using(
            using_expr,
            right_alias=right_alias,
            allowed_columns=allowed_columns,
        )
    return _join_condition_expr_from_on(
        on_expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _join_table_alias(table: exp.Table | None) -> str | None:
    if not isinstance(table, exp.Table):
        return None
    alias = table.alias_or_name
    if isinstance(alias, str) and alias:
        return alias
    name = table.name
    return name if name else None


def _join_condition_from_using(
    expr: exp.Expression | list[exp.Expression],
    *,
    right_alias: str | None,
    allowed_columns: frozenset[str],
) -> Expression:
    if not right_alias:
        msg = "USING clause requires a right table alias"
        raise DuckDBRelationQueryBuilderError(msg)
    columns = _using_column_names(expr)
    if not columns:
        msg = "USING clause requires at least one column"
        raise DuckDBRelationQueryBuilderError(msg)
    conditions: list[Expression] = []
    for column in columns:
        _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="join")
        left_expr = ColumnExpression(column)
        right_expr = ColumnExpression(f"{right_alias}.{column}")
        conditions.append(left_expr == right_expr)
    combined = conditions[0]
    for condition in conditions[1:]:
        combined &= condition
    return combined


def _using_column_names(expr: exp.Expression | list[exp.Expression]) -> tuple[str, ...]:
    values: list[str] = []
    if isinstance(expr, list):
        for item in expr:
            values.extend(_using_column_names(item))
        return tuple(values)
    if isinstance(expr, exp.Identifier):
        name = expr.this
        return (name,) if isinstance(name, str) and name else ()
    if isinstance(expr, exp.Column):
        return (_column_name(expr),)
    if isinstance(expr, exp.Tuple):
        for item in expr.expressions:
            values.extend(_using_column_names(item))
        return tuple(values)
    return ()


def _join_condition_expr_from_on(
    expr: exp.Expression | None,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if expr is None:
        msg = "JOIN requires an ON clause"
        raise DuckDBRelationQueryBuilderError(msg)
    if isinstance(expr, exp.Paren):
        return _join_condition_expr_from_on(
            expr.this,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.And):
        if expr.this is None or expr.expression is None:
            msg = "JOIN AND requires two expressions"
            raise DuckDBRelationQueryBuilderError(msg)
        return _join_condition_expr_from_on(
            expr.this,
            allowed_columns=allowed_columns,
            column_types=column_types,
        ) & _join_condition_expr_from_on(
            expr.expression,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.Or):
        if expr.this is None or expr.expression is None:
            msg = "JOIN OR requires two expressions"
            raise DuckDBRelationQueryBuilderError(msg)
        return _join_condition_expr_from_on(
            expr.this,
            allowed_columns=allowed_columns,
            column_types=column_types,
        ) | _join_condition_expr_from_on(
            expr.expression,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.Not):
        if expr.this is None:
            msg = "JOIN NOT requires an expression"
            raise DuckDBRelationQueryBuilderError(msg)
        return ~_join_condition_expr_from_on(
            expr.this,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, _AST_COMPARISON_TYPES):
        return _join_comparison_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    msg = f"Unsupported JOIN predicate: {type(expr).__name__}"
    raise DuckDBRelationQueryBuilderError(msg)


def _join_comparison_expr(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    left = expr.this
    right = expr.expression
    if left is None or right is None:
        msg = "JOIN comparisons require two expressions"
        raise DuckDBRelationQueryBuilderError(msg)
    left_expr = _join_operand_expr(
        left,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    right_expr = _join_operand_expr(
        right,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    if isinstance(expr, exp.EQ):
        return left_expr == right_expr
    if isinstance(expr, exp.NEQ):
        return left_expr != right_expr
    if isinstance(expr, exp.LT):
        return left_expr < right_expr
    if isinstance(expr, exp.LTE):
        return left_expr <= right_expr
    if isinstance(expr, exp.GT):
        return left_expr > right_expr
    if isinstance(expr, exp.GTE):
        return left_expr >= right_expr
    msg = f"Unsupported JOIN comparison: {type(expr).__name__}"
    raise DuckDBRelationQueryBuilderError(msg)


def _join_operand_expr(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if isinstance(expr, exp.Column):
        column = _column_name(expr)
        _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="join")
        return _qualified_column(expr)
    if isinstance(expr, (exp.Literal, exp.Boolean)):
        return ConstantExpression(_literal_value(expr))
    return _duckdb_expr_from_projection(
        expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _qualified_column(column: exp.Column) -> Expression:
    return ColumnExpression(_qualified_column_name(column))


def _qualified_column_name(column: exp.Column) -> str:
    name = _column_name(column)
    table = column.table
    if isinstance(table, exp.Identifier):
        qualifier = table.this
    elif isinstance(table, str):
        qualifier = table
    else:
        qualifier = None
    return f"{qualifier}.{name}" if qualifier else name


def _table_key_from_table(table: exp.Table) -> str:
    name = table.name
    schema = table.db
    if schema:
        return f"{schema}.{name}"
    return name


def _scan_dataset(
    *,
    con: DuckDBConnection,
    entry: DatasetManifestEntry,
    context: RelationBuildContext,
    filter_expression: ds.Expression | None,
    projection_columns: Sequence[str] | None,
) -> DuckDBRelation:
    schema = dataset_schema_for_entry(entry) or context.contract_schema
    scan_paths = _parquet_scan_paths(entry)
    hive_partitioning = bool(entry.manifest.partition_columns)
    try:
        return scan_parquet(
            con,
            scan_paths=scan_paths,
            hive_partitioning=hive_partitioning,
            union_by_name=True,
            columns=projection_columns,
        )
    except (duckdb.Error, TypeError, ValueError):
        pass
    dataset = dataset_for_entry(entry)
    options = DatasetScannerOptions(
        batch_size=context.scan_options.batch_size,
        batch_readahead=context.scan_options.batch_readahead,
        fragment_readahead=context.scan_options.fragment_readahead,
        filter_expression=filter_expression,
        use_threads=context.scan_options.use_threads,
        metrics_enabled=context.scan_options.metrics_enabled,
        schema=schema,
        columns=projection_columns,
        unify_schemas=context.scan_options.unify_schemas,
        schema_promote_options=context.scan_options.schema_promote_options,
    )
    scanner = dataset_scanner_for_entry(
        entry,
        options=options,
    )
    if schema is not None:
        reader = scanner.to_reader()
        aligned = align_reader_to_contract(
            reader,
            schema,
            extras_policy=extras_policy_from_schema(schema),
            schema_promote_options=context.scan_options.schema_promote_options,
        )
        try:
            return scan_arrow(con, source=aligned)
        except (duckdb.Error, TypeError, ValueError):
            return scan_arrow(con, source=scanner)
    try:
        return scan_arrow(con, source=scanner)
    except (duckdb.Error, TypeError, ValueError):
        reader = scanner.to_reader()
        try:
            return scan_arrow(con, source=reader)
        except (duckdb.Error, TypeError, ValueError):
            return scan_arrow(con, source=dataset)


def _parquet_scan_paths(entry: DatasetManifestEntry) -> list[str]:
    return [str(entry.dataset_dir)]


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


def _string_unary_projection(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression | None:
    if isinstance(expr, exp.Lower):
        return _duckdb_string_unary_expr(
            expr.this,
            func_name="lower",
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.Upper):
        return _duckdb_string_unary_expr(
            expr.this,
            func_name="upper",
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    return None


def _string_predicate_projection(
    expr: exp.Expression,
    *,
    context: _PredicateContext,
) -> Expression | None:
    if isinstance(expr, exp.Contains):
        return _duckdb_string_predicate_expr(
            expr.this,
            expr.expression,
            func_name="contains",
            context=context,
        )
    if isinstance(expr, exp.StartsWith):
        return _duckdb_string_predicate_expr(
            expr.this,
            expr.expression,
            func_name="starts_with",
            context=context,
        )
    return None


def _duckdb_expr_from_projection(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    result = _duckdb_projection_basic(
        expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    if result is not None:
        return result
    result = _duckdb_projection_structured(
        expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    if result is not None:
        return result
    result = _duckdb_projection_function(
        expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    if result is not None:
        return result
    result = _duckdb_projection_literal(expr)
    if result is not None:
        return result
    msg = f"Unsupported select expression: {type(expr).__name__}"
    raise DuckDBRelationQueryBuilderError(msg)


def _duckdb_projection_basic(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression | None:
    if isinstance(expr, exp.Star):
        _require_allowed_column(column="*", allowed_columns=allowed_columns, ctx="select")
        return ColumnExpression("*")
    if isinstance(expr, exp.Column):
        column = _column_name(expr)
        _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="select")
        return _cast_projection_column(
            ColumnExpression(_qualified_column_name(expr)),
            column=column,
            column_types=column_types,
        )
    if isinstance(expr, exp.Alias):
        return _duckdb_alias_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    return None


def _cast_projection_column(
    expr: Expression,
    *,
    column: str,
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if column_types is None:
        return expr
    column_type = column_types.get(column)
    if column_type is None:
        return expr
    duckdb_type = duckdb_type_for_column_type(column_type)
    if duckdb_type is None:
        return expr
    try:
        return expr.cast(duckdb_type)
    except (duckdb.Error, TypeError, ValueError):
        return expr


def _duckdb_projection_structured_core(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression | None:
    if isinstance(expr, exp.Coalesce):
        return _duckdb_coalesce_expr(
            expr.expressions,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.Cast):
        return _duckdb_cast_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.Interval):
        return _duckdb_interval_expr(expr)
    if isinstance(expr, exp.Case):
        return _duckdb_case_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    return None


def _duckdb_projection_structured_containers(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression | None:
    if isinstance(expr, exp.Array):
        return _duckdb_array_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.Map):
        return _duckdb_map_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.Struct):
        return _duckdb_struct_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    return None


def _duckdb_projection_structured_json(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression | None:
    if isinstance(expr, exp.JSONObject):
        return _duckdb_json_object_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.JSONArray):
        return _duckdb_json_array_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.Bracket):
        return _duckdb_bracket_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.JSONExtract):
        return _duckdb_json_extract_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    if isinstance(expr, exp.JSONExtractScalar):
        return _duckdb_json_extract_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    return None


def _duckdb_projection_structured_by_type(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression | None:
    result = _duckdb_projection_structured_core(
        expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    if result is not None:
        return result
    result = _duckdb_projection_structured_containers(
        expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    if result is not None:
        return result
    return _duckdb_projection_structured_json(
        expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _duckdb_projection_structured(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression | None:
    predicate_context = _PredicateContext(
        allowed_columns=allowed_columns,
        column_types=column_types,
        ctx="select",
    )
    result = _string_unary_projection(
        expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    if result is not None:
        return result
    result = _duckdb_projection_structured_by_type(
        expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    if result is not None:
        return result
    return _string_predicate_projection(expr, context=predicate_context)


def _duckdb_projection_function(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression | None:
    result: Expression | None = None
    if isinstance(expr, exp.DateAdd):
        result = _duckdb_date_add_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, exp.DateDiff):
        result = _duckdb_date_diff_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, exp.TimestampTrunc):
        result = _duckdb_date_trunc_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, exp.Extract):
        result = _duckdb_extract_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, exp.Anonymous):
        result = _duckdb_function_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    elif isinstance(expr, exp.Func):
        result = _duckdb_named_function_expr(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    return result


def _duckdb_projection_literal(expr: exp.Expression) -> Expression | None:
    if isinstance(expr, (exp.Boolean, exp.Literal)):
        return ConstantExpression(_literal_value(expr))
    if isinstance(expr, exp.Var):
        return _duckdb_var_expr(expr)
    return None


def _duckdb_alias_expr(
    expr: exp.Alias,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    inner = expr.this
    if inner is None:
        msg = "Alias requires an expression"
        raise DuckDBRelationQueryBuilderError(msg)
    alias = expr.alias
    value = _duckdb_expr_from_projection(
        inner,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    if isinstance(alias, str) and alias:
        return value.alias(alias)
    return value


def _duckdb_cast_expr(
    expr: exp.Cast,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    inner = expr.this
    if inner is None:
        msg = "CAST requires an expression"
        raise DuckDBRelationQueryBuilderError(msg)
    target = expr.args.get("to")
    if not isinstance(target, exp.DataType):
        msg = "CAST requires a target data type"
        raise DuckDBRelationQueryBuilderError(msg)
    duckdb_type = duckdb_type_for_column_type(target.sql(dialect="duckdb"))
    if duckdb_type is None:
        msg = f"Unsupported CAST target type: {target}"
        raise DuckDBRelationQueryBuilderError(msg)
    value = _duckdb_expr_from_projection(
        inner,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    return value.cast(duckdb_type)


def _duckdb_date_add_expr(
    expr: exp.DateAdd,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if expr.this is None or expr.expression is None:
        msg = "DATE_ADD requires a date expression and interval"
        raise DuckDBRelationQueryBuilderError(msg)
    unit_expr = expr.args.get("unit")
    if unit_expr is not None:
        interval_expr = _interval_expr_from_unit_value(expr.this, expr.expression)
        date_expr = _date_add_target_expr(
            unit_expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        return FunctionExpression("date_add", date_expr, interval_expr)
    date_expr = _duckdb_expr_from_projection(
        expr.this,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    interval_expr = _interval_expr_from_expression(
        expr.expression,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    return FunctionExpression("date_add", date_expr, interval_expr)


def _date_add_target_expr(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if isinstance(expr, exp.Var):
        name = expr.this
        if not isinstance(name, str) or not name:
            msg = "DATE_ADD target must be a column name"
            raise DuckDBRelationQueryBuilderError(msg)
        column = name if name in allowed_columns else name.lower()
        _require_allowed_column(column=column, allowed_columns=allowed_columns, ctx="date_add")
        return ColumnExpression(column)
    return _duckdb_expr_from_projection(
        expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _duckdb_date_diff_expr(
    expr: exp.DateDiff,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    unit_expr = expr.args.get("unit")
    if expr.this is None or expr.expression is None or unit_expr is None:
        msg = "DATE_DIFF requires a unit and two expressions"
        raise DuckDBRelationQueryBuilderError(msg)
    unit = _duckdb_expr_from_projection(
        unit_expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    start_expr = _duckdb_expr_from_projection(
        expr.expression,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    end_expr = _duckdb_expr_from_projection(
        expr.this,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    return FunctionExpression("date_diff", unit, start_expr, end_expr)


def _duckdb_date_trunc_expr(
    expr: exp.TimestampTrunc,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    unit_expr = expr.args.get("unit")
    if expr.this is None or unit_expr is None:
        msg = "DATE_TRUNC requires a unit and expression"
        raise DuckDBRelationQueryBuilderError(msg)
    unit = _duckdb_expr_from_projection(
        unit_expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    value_expr = _duckdb_expr_from_projection(
        expr.this,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    return FunctionExpression("date_trunc", unit, value_expr)


def _duckdb_extract_expr(
    expr: exp.Extract,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if expr.this is None or expr.expression is None:
        msg = "EXTRACT requires a unit and expression"
        raise DuckDBRelationQueryBuilderError(msg)
    unit = _duckdb_expr_from_projection(
        expr.this,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    value_expr = _duckdb_expr_from_projection(
        expr.expression,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    return FunctionExpression("date_part", unit, value_expr)


def _duckdb_interval_expr(expr: exp.Interval) -> Expression:
    if expr.this is None:
        msg = "INTERVAL requires a literal value"
        raise DuckDBRelationQueryBuilderError(msg)
    unit_expr = expr.args.get("unit")
    if unit_expr is None:
        msg = "INTERVAL requires a unit"
        raise DuckDBRelationQueryBuilderError(msg)
    interval = _interval_literal_from_parts(expr.this, unit_expr)
    return interval.cast(duckdb.sqltype("INTERVAL"))


def _interval_expr_from_expression(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if isinstance(expr, exp.Interval):
        return _duckdb_interval_expr(expr)
    if isinstance(expr, exp.Literal) and expr.is_string:
        return ConstantExpression(str(expr.this)).cast(duckdb.sqltype("INTERVAL"))
    return _duckdb_expr_from_projection(
        expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _interval_expr_from_unit_value(
    unit_expr: exp.Expression,
    value_expr: exp.Expression,
) -> Expression:
    interval = _interval_literal_from_parts(value_expr, unit_expr)
    return interval.cast(duckdb.sqltype("INTERVAL"))


def _interval_literal_from_parts(
    value_expr: exp.Expression,
    unit_expr: exp.Expression,
) -> Expression:
    unit = _interval_unit(unit_expr)
    value = _interval_value(value_expr)
    literal = f"{value} {unit}"
    return ConstantExpression(literal)


def _interval_unit(expr: exp.Expression) -> str:
    if isinstance(expr, exp.Var) or (isinstance(expr, exp.Literal) and expr.is_string):
        raw = expr.this
    else:
        msg = f"Unsupported interval unit: {type(expr).__name__}"
        raise DuckDBRelationQueryBuilderError(msg)
    if not isinstance(raw, str) or not raw:
        msg = "Interval unit must be a string"
        raise DuckDBRelationQueryBuilderError(msg)
    return raw.lower()


def _interval_value(expr: exp.Expression) -> str:
    value = _literal_value(expr)
    if isinstance(value, bool):
        msg = "Interval value must be numeric"
        raise DuckDBRelationQueryBuilderError(msg)
    if isinstance(value, (int, float, str)):
        return str(value)
    msg = f"Unsupported interval value: {type(value).__name__}"
    raise DuckDBRelationQueryBuilderError(msg)


def _duckdb_case_expr(
    expr: exp.Case,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    ifs = expr.args.get("ifs") or []
    if not ifs:
        msg = "CASE requires at least one WHEN clause"
        raise DuckDBRelationQueryBuilderError(msg)
    case_operand = expr.args.get("this")

    first = ifs[0]
    if not isinstance(first, exp.If):
        msg = "CASE WHEN clause is invalid"
        raise DuckDBRelationQueryBuilderError(msg)
    first_condition = _case_condition_expr(
        case_operand,
        first,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    first_value = _case_value_expr(
        first,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    case_expr = duckdb.CaseExpression(first_condition, first_value)
    for clause in ifs[1:]:
        if not isinstance(clause, exp.If):
            msg = "CASE WHEN clause is invalid"
            raise DuckDBRelationQueryBuilderError(msg)
        condition = _case_condition_expr(
            case_operand,
            clause,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        value = _case_value_expr(
            clause,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        case_expr = case_expr.when(condition, value)
    default_expr = expr.args.get("default")
    if default_expr is not None:
        default_value = _duckdb_expr_from_projection(
            default_expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        case_expr = case_expr.otherwise(default_value)
    return case_expr


def _duckdb_array_expr(
    expr: exp.Array,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    args = [
        _duckdb_expr_from_projection(
            item,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        for item in expr.expressions
    ]
    return FunctionExpression("list_value", *args)


def _duckdb_map_expr(
    expr: exp.Map,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    keys_expr = expr.args.get("keys")
    values_expr = expr.args.get("values")
    if keys_expr is None or values_expr is None:
        msg = "MAP requires both keys and values"
        raise DuckDBRelationQueryBuilderError(msg)
    keys = _duckdb_expr_from_projection(
        keys_expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    values = _duckdb_expr_from_projection(
        values_expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    return FunctionExpression("map", keys, values)


def _duckdb_struct_expr(
    expr: exp.Struct,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if not expr.expressions:
        msg = "STRUCT requires at least one field"
        raise DuckDBRelationQueryBuilderError(msg)
    args: list[Expression] = []
    for entry in expr.expressions:
        field_name, value_expr = _struct_field_entry(entry)
        value = _duckdb_expr_from_projection(
            value_expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        args.append(value.alias(field_name))
    return FunctionExpression("struct_pack", *args)


def _struct_field_entry(entry: exp.Expression) -> tuple[str, exp.Expression]:
    if isinstance(entry, exp.PropertyEQ):
        key_expr = entry.this
        value_expr = entry.expression
    elif isinstance(entry, exp.Alias):
        key_expr = entry.args.get("alias")
        value_expr = entry.this
    else:
        msg = f"Unsupported STRUCT field: {type(entry).__name__}"
        raise DuckDBRelationQueryBuilderError(msg)
    if value_expr is None:
        msg = "STRUCT field requires a value expression"
        raise DuckDBRelationQueryBuilderError(msg)
    name = _struct_field_name(key_expr)
    return name, value_expr


def _struct_field_name(expr: exp.Expression | None) -> str:
    if expr is None:
        msg = "STRUCT field requires a name"
        raise DuckDBRelationQueryBuilderError(msg)
    if isinstance(expr, exp.Identifier) or (isinstance(expr, exp.Literal) and expr.is_string):
        name = expr.this
    elif isinstance(expr, str):
        name = expr
    else:
        name = getattr(expr, "name", None)
    if not isinstance(name, str) or not name:
        msg = "STRUCT field name must be a non-empty string"
        raise DuckDBRelationQueryBuilderError(msg)
    return name


def _duckdb_json_object_expr(
    expr: exp.JSONObject,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if not expr.expressions:
        msg = "JSON_OBJECT requires at least one key/value pair"
        raise DuckDBRelationQueryBuilderError(msg)
    args: list[Expression] = []
    for item in expr.expressions:
        if not isinstance(item, exp.JSONKeyValue):
            msg = "JSON_OBJECT entries must be key/value pairs"
            raise DuckDBRelationQueryBuilderError(msg)
        key = _json_object_key(item.this)
        value_expr = item.expression
        if value_expr is None:
            msg = "JSON_OBJECT requires a value expression"
            raise DuckDBRelationQueryBuilderError(msg)
        args.append(ConstantExpression(key))
        args.append(
            _duckdb_expr_from_projection(
                value_expr,
                allowed_columns=allowed_columns,
                column_types=column_types,
            )
        )
    return FunctionExpression("json_object", *args)


def _json_object_key(expr: exp.Expression | None) -> str:
    if expr is None:
        msg = "JSON_OBJECT requires string keys"
        raise DuckDBRelationQueryBuilderError(msg)
    if isinstance(expr, exp.Identifier) or (isinstance(expr, exp.Literal) and expr.is_string):
        name = expr.this
    else:
        name = getattr(expr, "name", None)
    if not isinstance(name, str) or not name:
        msg = "JSON_OBJECT keys must be string literals"
        raise DuckDBRelationQueryBuilderError(msg)
    return name


def _duckdb_json_array_expr(
    expr: exp.JSONArray,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    args = [
        _duckdb_expr_from_projection(
            item,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        for item in expr.expressions
    ]
    return FunctionExpression("json_array", *args)


def _case_condition_expr(
    case_operand: exp.Expression | None,
    clause: exp.If,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    condition_expr = clause.this
    if condition_expr is None:
        msg = "CASE WHEN requires a condition"
        raise DuckDBRelationQueryBuilderError(msg)
    if case_operand is None:
        return _build_predicate_expr_ast(
            condition_expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    left = _duckdb_expr_from_projection(
        case_operand,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    right = _duckdb_expr_from_projection(
        condition_expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    return left == right


def _case_value_expr(
    clause: exp.If,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    value_expr = clause.args.get("true")
    if value_expr is None:
        msg = "CASE WHEN requires a THEN value"
        raise DuckDBRelationQueryBuilderError(msg)
    return _duckdb_expr_from_projection(
        value_expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )


def _duckdb_bracket_expr(
    expr: exp.Bracket,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    base_expr = _require_bracket_base(expr)
    index_value = _bracket_index_value(expr)
    base = _duckdb_expr_from_projection(
        base_expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    column_type = _column_type_for_base(base_expr, column_types=column_types)
    return _bracket_expression_for_type(
        base=base,
        index_value=index_value,
        column_type=column_type,
    )


def _require_bracket_base(expr: exp.Bracket) -> exp.Expression:
    base_expr = expr.this
    if base_expr is None:
        msg = "Bracket access requires an expression"
        raise DuckDBRelationQueryBuilderError(msg)
    return base_expr


def _bracket_index_value(expr: exp.Bracket) -> int | str:
    if len(expr.expressions) != 1:
        msg = "Bracket access requires a single index expression"
        raise DuckDBRelationQueryBuilderError(msg)
    index_expr = expr.expressions[0]
    index_value = _literal_value(index_expr)
    if not isinstance(index_value, (int, str)):
        msg = "Bracket index must be a string or integer literal"
        raise DuckDBRelationQueryBuilderError(msg)
    return index_value


def _column_type_for_base(
    base_expr: exp.Expression,
    *,
    column_types: Mapping[str, ColumnType] | None,
) -> ColumnType | None:
    if isinstance(base_expr, exp.Column) and column_types is not None:
        column = _column_name(base_expr)
        return column_types.get(column)
    return _base_type_hint(base_expr)


def _base_type_hint(expr: exp.Expression) -> ColumnType | None:
    if isinstance(expr, exp.Array):
        return "LIST"
    if isinstance(expr, exp.Map):
        return "MAP"
    if isinstance(expr, exp.Struct):
        return "STRUCT"
    if isinstance(
        expr,
        (
            exp.JSONExtract,
            exp.JSONExtractScalar,
            exp.JSONObject,
            exp.JSONArray,
            exp.ParseJSON,
        ),
    ):
        return "JSON"
    if isinstance(expr, exp.Anonymous):
        name = (expr.name or "").lower()
        if name in {"json", "json_array", "json_object"}:
            return "JSON"
    return None


def _bracket_expression_for_type(
    *,
    base: Expression,
    index_value: int | str,
    column_type: ColumnType | None,
) -> Expression:
    base_type = column_type_base(column_type) if column_type is not None else None
    if base_type == "LIST":
        return _list_bracket_expr(base=base, index_value=index_value)
    if base_type == "MAP":
        return _map_bracket_expr(base=base, index_value=index_value)
    if base_type == "STRUCT":
        return _struct_bracket_expr(base=base, index_value=index_value)
    if base_type == "JSON" or base_type is None:
        path = _json_path_from_bracket(index_value)
        return FunctionExpression("json_extract", base, ConstantExpression(path))
    msg = f"Unsupported bracket access for column type {column_type}"
    raise DuckDBRelationQueryBuilderError(msg)


def _list_bracket_expr(*, base: Expression, index_value: int | str) -> Expression:
    if not isinstance(index_value, int):
        msg = "LIST access requires an integer index"
        raise DuckDBRelationQueryBuilderError(msg)
    return FunctionExpression("list_extract", base, ConstantExpression(index_value))


def _map_bracket_expr(*, base: Expression, index_value: int | str) -> Expression:
    if not isinstance(index_value, str):
        msg = "MAP access requires a string key"
        raise DuckDBRelationQueryBuilderError(msg)
    return FunctionExpression("map_extract", base, ConstantExpression(index_value))


def _struct_bracket_expr(*, base: Expression, index_value: int | str) -> Expression:
    if not isinstance(index_value, str):
        msg = "STRUCT access requires a string key"
        raise DuckDBRelationQueryBuilderError(msg)
    return FunctionExpression("struct_extract", base, ConstantExpression(index_value))


def _duckdb_json_extract_expr(
    expr: exp.Expression,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if not isinstance(expr, (exp.JSONExtract, exp.JSONExtractScalar)):
        msg = "JSON extraction requires a JSONExtract expression"
        raise DuckDBRelationQueryBuilderError(msg)
    base_expr = expr.this
    path_expr = expr.expression
    if base_expr is None or path_expr is None:
        msg = "JSON extraction requires a base expression and path"
        raise DuckDBRelationQueryBuilderError(msg)
    base = _duckdb_expr_from_projection(
        base_expr,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    path_value = _json_path_literal(path_expr)
    func_name = "json_extract_string" if isinstance(expr, exp.JSONExtractScalar) else "json_extract"
    return FunctionExpression(func_name, base, ConstantExpression(path_value))


_JSON_PATH_QUOTE_MIN_LEN = 2


def _json_path_literal(expr: exp.Expression) -> str:
    if isinstance(expr, exp.JSONPath):
        rendered = expr.sql(dialect="duckdb")
        if (
            rendered.startswith("'")
            and rendered.endswith("'")
            and len(rendered) >= _JSON_PATH_QUOTE_MIN_LEN
        ):
            return rendered[1:-1]
        return rendered
    if isinstance(expr, exp.Literal) and expr.is_string:
        return str(expr.this)
    msg = f"Unsupported JSON path expression: {type(expr).__name__}"
    raise DuckDBRelationQueryBuilderError(msg)


def _json_path_from_bracket(value: int | str) -> str:
    if isinstance(value, int):
        return f"$[{value}]"
    return f"$.{value}"


def _duckdb_named_function_expr(
    expr: exp.Func,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    func_name = expr.sql_name().lower()
    func_name = _NAMED_FUNCTION_ALIASES.get(func_name, func_name)
    if func_name not in _NAMED_FUNCTIONS:
        msg = f"Unsupported function: {func_name or '<unknown>'}"
        raise DuckDBRelationQueryBuilderError(msg)
    args = [
        _duckdb_expr_from_projection(
            arg,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        for arg in _ordered_func_args(expr)
    ]
    return FunctionExpression(func_name, *args)


def _ordered_func_args(expr: exp.Expression) -> list[exp.Expression]:
    args: list[exp.Expression] = []
    for key in expr.arg_types:
        value = expr.args.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            args.extend(value)
        else:
            args.append(value)
    return args


def _duckdb_var_expr(expr: exp.Var) -> Expression:
    name = expr.this
    if not isinstance(name, str) or not name:
        msg = "Var expression requires a string value"
        raise DuckDBRelationQueryBuilderError(msg)
    return ConstantExpression(name.lower())


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
    if func_name in _GENERIC_FUNCTIONS:
        return _duckdb_generic_function(
            expr,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    msg = f"Unsupported function: {func_name or '<unknown>'}"
    raise DuckDBRelationQueryBuilderError(msg)


def _duckdb_generic_function(
    expr: exp.Anonymous,
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression:
    if not expr.expressions:
        msg = "Generic functions require at least one argument"
        raise DuckDBRelationQueryBuilderError(msg)
    args = [
        _duckdb_expr_from_projection(
            arg,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
        for arg in expr.expressions
    ]
    func_name = (expr.name or "").lower()
    func_name = _GENERIC_FUNCTION_ALIASES.get(func_name, func_name)
    return FunctionExpression(func_name, *args)


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
    value = _literal_value(value_expr)
    try:
        validate_filter_value(op=op, value=value, column_type=column_type)
    except FilterOpError as exc:
        raise DuckDBRelationQueryBuilderError(str(exc)) from exc
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
        items.append((_qualified_column_name(expr.this), bool(expr.args.get("desc"))))
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
_AST_COMPARISON_OPS: dict[type[exp.Expression], Op] = {
    exp.EQ: "eq",
    exp.NEQ: "ne",
    exp.LT: "lt",
    exp.LTE: "lte",
    exp.GT: "gt",
    exp.GTE: "gte",
}
_REVERSED_OPS: dict[Op, Op] = {
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
    value = _literal_value(value_expr)
    return _build_string_predicate(
        col_expr=ColumnExpression(column),
        op=op,
        value=value,
        column_type=column_type,
    )


def _is_literal(expr: exp.Expression | None) -> bool:
    return isinstance(expr, (exp.Literal, exp.Boolean))


def _literal_value(expr: exp.Expression | None) -> FilterScalar:
    if expr is None:
        msg = "Expected literal value"
        raise DuckDBRelationQueryBuilderError(msg)
    for extractor in (
        _literal_from_to_py,
        _literal_from_neg,
        _literal_from_boolean,
        _literal_from_literal,
    ):
        value = extractor(expr)
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


def _literal_from_neg(expr: exp.Expression) -> FilterScalar | None:
    result: FilterScalar | None = None
    if isinstance(expr, exp.Neg):
        inner = expr.this
        if inner is not None:
            value = _literal_from_literal(inner)
            if isinstance(value, (int, float)):
                result = -value
    return result


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


_STRING_FUNC_ARG_COUNT = 2
_STRING_FUNC_MAP: dict[str, Op] = {
    "contains": "contains",
    "starts_with": "startswith",
}
_STRING_PREDICATE_FUNCS = frozenset(_STRING_FUNC_MAP.keys())
_STRING_UNARY_FUNCS = frozenset({"lower", "upper"})
_NAMED_FUNCTION_ALIASES = {
    "array_size": "array_length",
    "dateadd": "date_add",
    "datediff": "date_diff",
    "extract": "date_part",
    "json_extract_scalar": "json_extract_string",
    "parse_json": "json",
    "timestamp_trunc": "date_trunc",
}
_GENERIC_FUNCTION_ALIASES = {
    "array_size": "array_length",
    "json_extract_scalar": "json_extract_string",
    "list_length": "array_length",
    "parse_json": "json",
}
_GENERIC_FUNCTIONS = frozenset(
    {
        "array_length",
        "date_add",
        "date_diff",
        "date_part",
        "date_sub",
        "date_trunc",
        "json",
        "json_array",
        "json_extract",
        "json_extract_scalar",
        "json_extract_string",
        "json_object",
        "list_extract",
        "list_value",
        "map",
        "map_extract",
        "map_keys",
        "map_values",
        "struct_pack",
        "struct_extract",
    }
)
_NAMED_FUNCTIONS = _GENERIC_FUNCTIONS


def _duckdb_type_for_column(column_type: ColumnType | None) -> DuckDBPyType | None:
    if column_type is None:
        return None
    mapping = complex_type_mapping(column_type)
    if mapping is not None and mapping.duckdb_type is not None:
        return mapping.duckdb_type
    return duckdb_type_for_column_type(column_type)


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
    op: Op,
    value: FilterValue,
    column_type: ColumnType | None,
) -> Expression:
    try:
        validated = validate_filter_value(op=op, value=value, column_type=column_type)
    except FilterOpError as exc:
        raise DuckDBRelationQueryBuilderError(str(exc)) from exc
    if isinstance(validated, list):
        msg = f"{op} operator does not support list value"
        raise DuckDBRelationQueryBuilderError(msg)
    literal = _typed_constant(validated, column_type=column_type)
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
    try:
        validated = validate_filter_value(op="in", value=value, column_type=column_type)
    except FilterOpError as exc:
        raise DuckDBRelationQueryBuilderError(str(exc)) from exc
    if not isinstance(validated, list):
        msg = "IN operator requires list value"
        raise DuckDBRelationQueryBuilderError(msg)
    if not validated:
        return ConstantExpression(0) == ConstantExpression(1)
    constants = [_typed_constant(item, column_type=column_type) for item in validated]
    return col_expr.isin(*constants)


def _build_string_predicate(
    *,
    col_expr: Expression,
    op: Op,
    value: FilterValue,
    column_type: ColumnType | None,
) -> Expression:
    try:
        validated = validate_filter_value(op=op, value=value, column_type=column_type)
    except FilterOpError as exc:
        raise DuckDBRelationQueryBuilderError(str(exc)) from exc
    if not isinstance(validated, str):
        msg = f"{op} operator requires string value"
        raise DuckDBRelationQueryBuilderError(msg)
    literal = _typed_constant(validated, column_type=column_type)
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
    "RelationBuildContext",
    "RelationScanOptions",
    "apply_query_ast",
    "build_relation_plan",
    "validate_query_ast",
]
