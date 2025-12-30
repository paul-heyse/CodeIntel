"""Tests for tombstone SQLGlot filter helpers."""

from __future__ import annotations

import pytest
from sqlglot import exp

from codeintel.serving.semantic.tombstones import apply_tombstone_filter

pytestmark = pytest.mark.no_runtime_env


def test_apply_tombstone_filter_adds_not_exists() -> None:
    """Tombstone filter adds NOT EXISTS with tombstone table."""
    ast = exp.select("id").from_("docs.symbols")
    filtered = apply_tombstone_filter(
        ast,
        table_key="docs.symbols",
        primary_key=["id"],
        snapshot_id=42,
    )
    sql = filtered.sql(dialect="duckdb")
    assert "__tombstones" in sql
    assert "NOT EXISTS" in sql


def test_apply_tombstone_filter_skips_join_queries() -> None:
    """Tombstone filter leaves join queries unchanged."""
    base = exp.Table(this=exp.to_identifier("symbols"), db=exp.to_identifier("docs"))
    join_table = exp.Table(this=exp.to_identifier("modules"), db=exp.to_identifier("docs"))
    ast = exp.select("symbols.id").from_(base).join(join_table, on="symbols.id = modules.id")
    filtered = apply_tombstone_filter(
        ast,
        table_key="docs.symbols",
        primary_key=["id"],
        snapshot_id=7,
    )
    assert filtered.sql(dialect="duckdb") == ast.sql(dialect="duckdb")
