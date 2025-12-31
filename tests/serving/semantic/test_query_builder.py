"""Tests for the DuckDB relation query builder."""

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


def test_relation_ast_builder_filters_orders_and_paginates() -> None:
    """Relation AST builder applies filters, ordering, and pagination."""
    con = duckdb.connect()
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.v_demo (id INTEGER, label VARCHAR)")
    con.executemany(
        "INSERT INTO docs.v_demo VALUES (?, ?)",
        [(1, "one"), (2, "two"), (3, "three")],
    )
    spec = SemanticQuerySpec(
        view_id="demo.view",
        table_key="docs.v_demo",
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
    result = apply_query_ast(
        con.sql("SELECT * FROM docs.v_demo"),
        ast=ast,
        allowed_columns=spec.allowed_columns,
        column_types=spec.column_types,
    ).fetchall()
    expect_equal(result, [(3, "three"), (2, "two")])


def test_relation_ast_builder_supports_alias_and_lower() -> None:
    """AST builder supports projection aliases and lower()."""
    con = duckdb.connect()
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.v_demo (label VARCHAR)")
    con.executemany(
        "INSERT INTO docs.v_demo VALUES (?)",
        [("One",), ("Two",)],
    )
    ast = parse_one(
        "SELECT lower(label) AS label_lower FROM docs.v_demo",
        dialect="duckdb",
    )
    result = apply_query_ast(
        con.sql("SELECT * FROM docs.v_demo"),
        ast=ast,
        allowed_columns=frozenset({"label"}),
        column_types={"label": "VARCHAR"},
    ).fetchall()
    expect_equal(result, [("one",), ("two",)])


def test_relation_ast_builder_rejects_unknown_column() -> None:
    """AST builder rejects unknown columns."""
    con = duckdb.connect()
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.v_demo (id INTEGER)")
    spec = SemanticQuerySpec(
        view_id="demo.view",
        table_key="docs.v_demo",
        allowed_columns=frozenset({"id"}),
        columns=["id"],
        filters=[],
        order_by=[],
        limit=10,
        offset=0,
    )
    ast = build_sqlglot_query(spec=spec, allowed_columns=spec.allowed_columns)
    with pytest.raises(DuckDBRelationQueryBuilderError, match="Unknown select column"):
        apply_query_ast(con.sql("SELECT * FROM docs.v_demo"), ast=ast, allowed_columns=frozenset())


def test_relation_ast_builder_validates_operators() -> None:
    """AST builder rejects unsupported operators."""
    con = duckdb.connect()
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.v_demo (id INTEGER)")
    table = exp.Table(this=exp.to_identifier("v_demo"), db=exp.to_identifier("docs"))
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
            con.sql("SELECT * FROM docs.v_demo"),
            ast=ast,
            allowed_columns=frozenset({"id"}),
            column_types={"id": "INTEGER"},
        )
