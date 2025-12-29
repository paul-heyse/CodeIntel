"""Tests for DuckDB relation builder AST handling."""

from __future__ import annotations

import duckdb
import pytest
from sqlglot import exp, parse_one

from codeintel.serving.semantic.duckdb_relation_builder import (
    DuckDBRelationQueryBuilderError,
    apply_query_ast,
)
from codeintel.serving.semantic.models import FilterSpec
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.serving.semantic.sqlglot_query_builder import build_sqlglot_query
from tests._helpers.assertions.expectation_assertions import expect_equal

pytestmark = pytest.mark.no_runtime_env


def _demo_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE demo (id INTEGER, label VARCHAR)")
    con.execute("INSERT INTO demo VALUES (1, 'one'), (2, 'two'), (3, 'three')")
    return con


def test_duckdb_ast_builder_filters_orders_and_paginates() -> None:
    """DuckDB AST builder applies filters, ordering, and pagination."""
    con = _demo_connection()
    try:
        relation = con.table("demo")
        spec = SemanticQuerySpec(
            view_id="demo.view",
            table_key="main.demo",
            allowed_columns=frozenset({"id", "label"}),
            columns=["id", "label"],
            filters=[FilterSpec(column="id", op="gte", value=2)],
            order_by=["-id"],
            limit=10,
            offset=0,
            column_types={"id": "INTEGER", "label": "VARCHAR"},
        )
        ast = build_sqlglot_query(
            spec=spec,
            allowed_columns=spec.allowed_columns,
            column_types=spec.column_types,
        )
        rows = apply_query_ast(
            relation,
            ast=ast,
            allowed_columns=spec.allowed_columns,
            column_types=spec.column_types,
        ).fetchall()
        expect_equal(rows, expected=[(3, "three"), (2, "two")])
    finally:
        con.close()


def test_duckdb_ast_builder_supports_alias_and_lower() -> None:
    """DuckDB AST builder supports projection aliases and lower()."""
    con = _demo_connection()
    try:
        relation = con.table("demo")
        ast = parse_one(
            "SELECT lower(label) AS label_lower FROM main.demo",
            dialect="duckdb",
        )
        rows = apply_query_ast(
            relation,
            ast=ast,
            allowed_columns=frozenset({"label"}),
            column_types={"label": "VARCHAR"},
        ).fetchall()
        expect_equal(rows, expected=[("one",), ("two",), ("three",)])
    finally:
        con.close()


def test_duckdb_ast_builder_rejects_unknown_column() -> None:
    """Unknown columns in AST selections are rejected."""
    con = _demo_connection()
    try:
        relation = con.table("demo")
        ast = exp.select(exp.Column(this=exp.to_identifier("oops")))
        with pytest.raises(DuckDBRelationQueryBuilderError, match="Unknown select column"):
            apply_query_ast(
                relation,
                ast=ast,
                allowed_columns=frozenset({"id"}),
                column_types={"id": "INTEGER"},
            )
    finally:
        con.close()


def test_duckdb_ast_builder_rejects_invalid_operator() -> None:
    """Invalid operator usage should raise a builder error."""
    con = _demo_connection()
    try:
        relation = con.table("demo")
        table = exp.Table(this=exp.to_identifier("demo"), db=exp.to_identifier("main"))
        predicate = exp.Anonymous(
            this="contains",
            expressions=[
                exp.Column(this=exp.to_identifier("id")),
                exp.Literal.string("oops"),
            ],
        )
        ast = exp.select(exp.Column(this=exp.to_identifier("id"))).from_(table).where(predicate)
        with pytest.raises(
            DuckDBRelationQueryBuilderError,
            match="Operator contains is not supported for column type INTEGER",
        ):
            apply_query_ast(
                relation,
                ast=ast,
                allowed_columns=frozenset({"id"}),
                column_types={"id": "INTEGER"},
            )
    finally:
        con.close()
