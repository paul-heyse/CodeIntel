"""Tests for safe query building."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import ibis
import pandas as pd
import pytest

from codeintel.serving.semantic.models import FilterSpec
from codeintel.serving.semantic.query_builder import (
    QueryBuilderError,
    SemanticQueryPlan,
    build_query,
)
from tests._helpers.assertions.expectation_assertions import expect_equal

if TYPE_CHECKING:
    from pathlib import Path


def _make_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.v_demo (id INTEGER, label VARCHAR)")
    con.execute("INSERT INTO docs.v_demo VALUES (1, 'one'), (2, 'two'), (3, 'three')")
    return con


def _build_query_from_memory(plan: SemanticQueryPlan) -> None:
    con = duckdb.connect(database=":memory:")
    try:
        ibis_con = ibis.duckdb.from_connection(con)
        built = build_query(ibis_con=ibis_con, plan=plan)
        for table_name in built.temp_tables:
            con.unregister(table_name)
    finally:
        con.close()


def _build_query_from_file(db_path: Path, plan: SemanticQueryPlan) -> None:
    con = _make_db(db_path)
    try:
        ibis_con = ibis.duckdb.from_connection(con)
        built = build_query(ibis_con=ibis_con, plan=plan)
        for table_name in built.temp_tables:
            con.unregister(table_name)
    finally:
        con.close()


def test_build_query_filters_orders_and_paginates(tmp_path: Path) -> None:
    """Build query applies filters, ordering, and pagination."""
    con = _make_db(tmp_path / "demo.duckdb")
    try:
        ibis_con = ibis.duckdb.from_connection(con)
        built = build_query(
            ibis_con=ibis_con,
            plan=SemanticQueryPlan(
                table_key="docs.v_demo",
                columns=["id", "label"],
                allowed_columns=frozenset({"id", "label"}),
                filters=[FilterSpec(column="id", op="gte", value=2)],
                order_by=["-id"],
                limit=10,
                offset=0,
            ),
        )
        df = pd.DataFrame(built.expr.execute(params=built.execute_params()))
        rows = df.to_dict(orient="records")
        expect_equal(rows, [{"id": 3, "label": "three"}, {"id": 2, "label": "two"}])
        for table_name in built.temp_tables:
            con.unregister(table_name)
    finally:
        con.close()


def test_build_query_rejects_unknown_column() -> None:
    """Unknown columns are rejected."""
    plan = SemanticQueryPlan(
        table_key="docs.v_demo",
        columns=["id", "oops"],
        allowed_columns=frozenset({"id"}),
        filters=[],
        order_by=[],
        limit=10,
        offset=0,
    )
    with pytest.raises(QueryBuilderError, match="Unknown column"):
        _build_query_from_memory(plan)


def test_build_query_validates_operators(tmp_path: Path) -> None:
    """Unsupported operators error out."""
    plan = SemanticQueryPlan(
        table_key="docs.v_demo",
        columns=["id"],
        allowed_columns=frozenset({"id"}),
        filters=[FilterSpec(column="id", op="contains", value=123)],
        order_by=[],
        limit=10,
        offset=0,
    )
    with pytest.raises(QueryBuilderError, match="requires string value"):
        _build_query_from_file(tmp_path / "demo.duckdb", plan)
