"""DuckDB relation-based query builder for semantic specs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.serving.semantic.filter_ops import allowed_ops_for_column_type
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

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.semantic.datasets import DatasetManifestEntry, DatasetManifestIndex


class DuckDBRelationQueryBuilderError(ValueError):
    """Raised when a relation-based query cannot be built."""


def build_relation_plan(
    *,
    con: DuckDBConnection,
    spec: SemanticQuerySpec,
    dataset_manifests: DatasetManifestIndex,
    column_types: Mapping[str, ColumnType] | None = None,
) -> DuckDBRelation:
    """Build a DuckDB relation plan for a semantic query spec.

    Returns
    -------
    DuckDBRelation
        Lazy relation representing the query plan.
    """
    relation = _resolve_relation(con=con, table_key=spec.table_key, manifests=dataset_manifests)
    return apply_query_spec(
        relation,
        spec=spec,
        allowed_columns=spec.allowed_columns,
        column_types=column_types,
    )


def apply_query_spec(
    relation: DuckDBRelation,
    *,
    spec: SemanticQuerySpec,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None = None,
) -> DuckDBRelation:
    """Apply a semantic query spec to a DuckDB relation.

    Returns
    -------
    DuckDBRelation
        Updated relation reflecting the applied filters, ordering, and limits.
    """
    _validate_pagination(limit=spec.limit, offset=spec.offset)

    for col in spec.columns:
        _require_allowed_column(column=col, allowed_columns=allowed_columns, ctx="select")

    predicates = _build_predicates(
        filters=spec.filters,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    if predicates is not None:
        relation = relation.filter(predicates)

    if spec.order_by:
        relation = relation.order(_order_by_expr(spec.order_by, allowed_columns=allowed_columns))

    relation = relation.select(*spec.columns)

    if spec.limit or spec.offset:
        relation = relation.limit(spec.limit, offset=spec.offset)

    return relation


def _resolve_relation(
    *,
    con: DuckDBConnection,
    table_key: str,
    manifests: DatasetManifestIndex,
) -> DuckDBRelation:
    entry = manifests.get(table_key)
    if entry is not None:
        return _scan_dataset(con=con, entry=entry)
    try:
        return con.table(table_key)
    except DuckDBCatalogException as exc:
        msg = f"Unknown DuckDB table/view: {table_key}"
        raise DuckDBRelationQueryBuilderError(msg) from exc


def _scan_dataset(*, con: DuckDBConnection, entry: DatasetManifestEntry) -> DuckDBRelation:
    hive_partitioning = bool(entry.manifest.partition_columns)
    if entry.manifest.files:
        paths = [str(entry.dataset_dir / path) for path in entry.manifest.files]
        return con.read_parquet(paths, hive_partitioning=hive_partitioning, union_by_name=True)
    glob = str(entry.dataset_dir / "**" / "*.parquet")
    return con.read_parquet(glob, hive_partitioning=hive_partitioning, union_by_name=True)


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


def _build_predicates(
    *,
    filters: list[FilterSpec],
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> Expression | None:
    if not filters:
        return None
    predicates: list[Expression] = []
    for filt in filters:
        _require_allowed_column(
            column=filt.column,
            allowed_columns=allowed_columns,
            ctx="filter",
        )
        predicates.append(_build_predicate(filt=filt, column_types=column_types))
    combined = _combine_predicates(predicates)
    if combined is None:
        return None
    return combined


def _build_predicate(
    *, filt: FilterSpec, column_types: Mapping[str, ColumnType] | None
) -> Expression:
    column_type = column_types.get(filt.column) if column_types is not None else None
    allowed_ops = allowed_ops_for_column_type(column_type)
    if filt.op not in allowed_ops:
        msg = (
            "Operator "
            f"{filt.op} is not supported for column type {column_type or _UNKNOWN_COLUMN_TYPE}"
        )
        raise DuckDBRelationQueryBuilderError(msg)

    col_expr = ColumnExpression(filt.column)
    op = filt.op
    value = filt.value

    if op in _COMPARISON_OPS:
        return _build_comparison_predicate(
            col_expr=col_expr,
            op=op,
            value=value,
            column_type=column_type,
        )
    if op == "in":
        return _build_in_predicate(col_expr=col_expr, value=value, column_type=column_type)
    if op in _STRING_OPS:
        return _build_string_predicate(
            col_expr=col_expr,
            op=op,
            value=value,
            column_type=column_type,
        )

    msg = f"Unsupported operator: {op}"
    raise DuckDBRelationQueryBuilderError(msg)


_UNKNOWN_COLUMN_TYPE = "UNKNOWN"
_COMPARISON_OPS = frozenset({"eq", "ne", "lt", "lte", "gt", "gte"})
_ORDERING_OPS = frozenset({"lt", "lte", "gt", "gte"})
_STRING_OPS = frozenset({"contains", "startswith"})


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
    if op in _ORDERING_OPS and column_type == "VARCHAR":
        msg = f"Operator {op} is not supported for string columns"
        raise DuckDBRelationQueryBuilderError(msg)
    literal = ConstantExpression(value)
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
    if column_type == "JSON":
        msg = "IN operator is not supported for JSON columns"
        raise DuckDBRelationQueryBuilderError(msg)
    constants = [ConstantExpression(item) for item in value]
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
    if column_type is not None and column_type != "VARCHAR":
        msg = f"{op} operator is only supported for VARCHAR columns"
        raise DuckDBRelationQueryBuilderError(msg)
    literal = ConstantExpression(value)
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


def _combine_predicates(predicates: Sequence[Expression]) -> Expression | None:
    if not predicates:
        return None
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined &= predicate
    return combined


__all__ = ["DuckDBRelationQueryBuilderError", "apply_query_spec", "build_relation_plan"]
