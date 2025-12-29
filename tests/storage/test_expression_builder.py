"""Tests for DuckDB expression helpers."""

from __future__ import annotations

import pytest

from codeintel.storage.duckdb_types import Expression
from codeintel.storage.queries.expressions import and_all, col, eq, lit, snapshot_filter

pytestmark = pytest.mark.no_runtime_env


def test_and_all_requires_expressions() -> None:
    """Ensure and_all enforces non-empty expressions."""
    with pytest.raises(ValueError, match="and_all requires at least one expression"):
        and_all([])


def test_expression_helpers_return_expressions() -> None:
    """Ensure expression helpers return DuckDB expressions."""
    expr = col("repo") == lit("demo")
    assert isinstance(expr, Expression)

    combined = and_all([expr, eq("commit", "abc")])
    assert isinstance(combined, Expression)

    snapshot = snapshot_filter(repo="demo", commit="abc")
    assert isinstance(snapshot, Expression)
