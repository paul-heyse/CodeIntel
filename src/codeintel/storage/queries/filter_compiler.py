"""Unified filter compiler for SQLGlot, DuckDB, Arrow, and Polars (core re-export)."""

from __future__ import annotations

from codeintel.core.queries.filter_compiler import (
    FilterCompilerError,
    FilterPredicate,
    arrow_filter_expression,
    compile_filter_predicates,
    duckdb_filter_expression,
    polars_filter_expression,
    sqlglot_filter_expression,
)

__all__ = [
    "FilterCompilerError",
    "FilterPredicate",
    "arrow_filter_expression",
    "compile_filter_predicates",
    "duckdb_filter_expression",
    "polars_filter_expression",
    "sqlglot_filter_expression",
]
