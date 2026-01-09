"""Unified filter compiler for SQLGlot, DuckDB, Arrow, and Polars (core re-export)."""

from __future__ import annotations

from codeintel.core.queries.filter_compiler import (
    FilterCompilerError,
    FilterPredicate,
    QuerySpecFilterRequest,
    arrow_filter_expression,
    arrow_predicate_from_filters,
    compile_filter_predicates,
    duckdb_filter_expression,
    polars_filter_expression,
    queryspec_from_filters,
    sqlglot_filter_expression,
)

__all__ = [
    "FilterCompilerError",
    "FilterPredicate",
    "QuerySpecFilterRequest",
    "arrow_filter_expression",
    "arrow_predicate_from_filters",
    "compile_filter_predicates",
    "duckdb_filter_expression",
    "polars_filter_expression",
    "queryspec_from_filters",
    "sqlglot_filter_expression",
]
