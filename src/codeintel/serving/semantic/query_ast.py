"""SQLGlot-backed query AST helpers for semantic serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.columnar.queryspec import QuerySpec, projection_spec_from_columns
from codeintel.core.sqlglot_tools import canonicalize_select_duckdb, schema_mapping_for_table_key
from codeintel.serving.semantic.arrow_plan_builder import ArrowPlanSpec, build_arrow_plan_spec
from codeintel.serving.semantic.duckdb_relation_builder import validate_query_ast
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.serving.semantic.sqlglot_query_builder import build_sqlglot_query
from codeintel.storage.datasets.manifest_index import dataset_filter_expression

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlglot import exp


@dataclass(frozen=True, slots=True)
class ServingQuery:
    """Semantic query bundle with SQLGlot AST."""

    spec: SemanticQuerySpec
    ast: exp.Select
    arrow_plan: ArrowPlanSpec | None
    query_spec: QuerySpec


_ALLOWED_ANONYMOUS_FUNCTIONS = frozenset(
    {
        "contains",
        "starts_with",
        "date_add",
        "date_diff",
        "date_part",
        "date_sub",
        "date_trunc",
        "json_extract",
        "json_extract_scalar",
        "list_extract",
        "list_value",
        "map_extract",
        "map_keys",
        "map_values",
        "struct_pack",
        "struct_extract",
    }
)


def normalize_serving_ast(
    ast: exp.Select,
    *,
    schema: Mapping[str, Mapping[str, str]] | None = None,
) -> exp.Select:
    """Normalize a serving AST for deterministic fingerprints.

    Parameters
    ----------
    ast
        SQLGlot Select expression to normalize.
    schema
        Optional SQLGlot schema mapping for type-aware canonicalization.

    Returns
    -------
    sqlglot.expressions.Select
        Canonicalized Select expression.
    """
    return canonicalize_select_duckdb(ast, schema=schema)


def build_serving_query(*, spec: SemanticQuerySpec) -> ServingQuery:
    """Build a ServingQuery from a semantic spec.

    Returns
    -------
    ServingQuery
        Serving query bundle with SQLGlot AST.
    """
    ast = build_sqlglot_query(
        spec=spec,
        allowed_anonymous_functions=_ALLOWED_ANONYMOUS_FUNCTIONS,
        allow_aggregates=False,
        log_context="serving_query_ast",
    )
    schema_mapping = schema_mapping_for_table_key(
        spec.table_key,
        column_types=spec.column_types,
    )
    canonical = normalize_serving_ast(ast, schema=schema_mapping)
    validate_query_ast(
        ast=canonical,
        allowed_columns=spec.allowed_columns,
        column_types=spec.column_types,
    )
    filter_expression = dataset_filter_expression(
        filters=spec.filters,
        allowed_columns=spec.allowed_columns,
        column_types=spec.column_types,
    )
    projection = projection_spec_from_columns(_plan_columns_for_spec(spec))
    query_spec = QuerySpec(
        predicate=filter_expression,
        pushdown_predicate=filter_expression,
        projection=projection,
    )
    arrow_plan = build_arrow_plan_spec(spec=spec, ast=canonical)
    return ServingQuery(
        spec=spec,
        ast=canonical,
        arrow_plan=arrow_plan,
        query_spec=query_spec,
    )


def _plan_columns_for_spec(spec: SemanticQuerySpec) -> tuple[str, ...]:
    columns = set(spec.columns)
    for filt in spec.filters:
        columns.add(filt.column)
    for order in spec.order_by:
        column = order[1:] if order.startswith("-") else order
        columns.add(column)
    if not columns:
        return tuple(spec.columns)
    return tuple(sorted(columns))


__all__ = ["ServingQuery", "build_serving_query", "normalize_serving_ast"]
