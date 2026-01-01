"""Tests for shared filter compiler helpers."""

from __future__ import annotations

import pytest

from codeintel.core.filters import FilterSpecInput
from codeintel.storage.duckdb_types import Expression
from codeintel.storage.queries.filter_compiler import (
    FilterCompilerError,
    compile_filter_predicates,
    duckdb_filter_expression,
)

pytestmark = pytest.mark.no_runtime_env


def test_compile_filter_predicates_rejects_unknown_column() -> None:
    """Ensure filter compilation fails when columns are not allowed."""
    filters = [FilterSpecInput(column="repo", op="eq", value="demo")]
    with pytest.raises(FilterCompilerError, match="Unknown filter column"):
        compile_filter_predicates(filters, allowed_columns=frozenset({"commit"}))


def test_duckdb_filter_expression_builds_expression() -> None:
    """Ensure filter compilation emits DuckDB expressions."""
    filters = [
        FilterSpecInput(column="repo", op="eq", value="demo"),
        FilterSpecInput(column="commit", op="eq", value="abc"),
    ]
    predicates = compile_filter_predicates(
        filters,
        allowed_columns=frozenset({"commit", "repo"}),
    )
    expression = duckdb_filter_expression(predicates)
    assert isinstance(expression, Expression)
