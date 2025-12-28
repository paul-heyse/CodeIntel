"""Tests for the Polars semantic query builder."""

from __future__ import annotations

import pytest
from sqlglot import exp

from codeintel.serving.semantic.models import FilterSpec
from codeintel.serving.semantic.polars_query_builder import (
    PolarsQueryBuilderError,
    apply_query_ast,
    apply_query_spec,
)
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.serving.semantic.sqlglot_query_builder import build_sqlglot_query
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_polars_query_builder_filters_orders_and_paginates() -> None:
    """Polars query builder applies filters, ordering, and pagination."""
    pl = pytest.importorskip("polars")
    lazy = pl.DataFrame({"id": [1, 2, 3], "label": ["one", "two", "three"]}).lazy()
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
    result = apply_query_spec(
        lazy,
        spec=spec,
        allowed_columns=spec.allowed_columns,
        column_types=spec.column_types,
    ).collect()
    expect_equal(result.to_dicts(), [{"id": 3, "label": "three"}, {"id": 2, "label": "two"}])


def test_polars_query_builder_rejects_unknown_column() -> None:
    """Unknown columns are rejected."""
    pl = pytest.importorskip("polars")
    lazy = pl.DataFrame({"id": [1]}).lazy()
    spec = SemanticQuerySpec(
        view_id="demo.view",
        table_key="docs.v_demo",
        allowed_columns=frozenset({"id"}),
        columns=["id", "oops"],
        filters=[],
        order_by=[],
        limit=10,
        offset=0,
    )
    with pytest.raises(PolarsQueryBuilderError, match="Unknown select column"):
        apply_query_spec(lazy, spec=spec, allowed_columns=spec.allowed_columns)


def test_polars_query_builder_validates_operators() -> None:
    """Unsupported operators error out."""
    pl = pytest.importorskip("polars")
    lazy = pl.DataFrame({"id": [1]}).lazy()
    spec = SemanticQuerySpec(
        view_id="demo.view",
        table_key="docs.v_demo",
        allowed_columns=frozenset({"id"}),
        columns=["id"],
        filters=[FilterSpec(column="id", op="contains", value=123)],
        order_by=[],
        limit=10,
        offset=0,
        column_types={"id": "INTEGER"},
    )
    with pytest.raises(
        PolarsQueryBuilderError,
        match="Operator contains is not supported for column type INTEGER",
    ):
        apply_query_spec(
            lazy,
            spec=spec,
            allowed_columns=spec.allowed_columns,
            column_types=spec.column_types,
        )


def test_polars_ast_builder_filters_orders_and_paginates() -> None:
    """Polars AST builder applies filters, ordering, and pagination."""
    pl = pytest.importorskip("polars")
    lazy = pl.DataFrame({"id": [1, 2, 3], "label": ["one", "two", "three"]}).lazy()
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
        lazy,
        ast=ast,
        allowed_columns=spec.allowed_columns,
        column_types=spec.column_types,
    ).collect()
    expect_equal(result.to_dicts(), [{"id": 3, "label": "three"}, {"id": 2, "label": "two"}])


def test_polars_ast_builder_rejects_unknown_column() -> None:
    """AST builder rejects unknown columns."""
    pl = pytest.importorskip("polars")
    lazy = pl.DataFrame({"id": [1]}).lazy()
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
    with pytest.raises(PolarsQueryBuilderError, match="Unknown select column"):
        apply_query_ast(lazy, ast=ast, allowed_columns=frozenset())


def test_polars_ast_builder_validates_operators() -> None:
    """AST builder rejects unsupported operators."""
    pl = pytest.importorskip("polars")
    lazy = pl.DataFrame({"id": [1]}).lazy()
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
        PolarsQueryBuilderError,
        match="Operator contains is not supported for column type INTEGER",
    ):
        apply_query_ast(
            lazy,
            ast=ast,
            allowed_columns=frozenset({"id"}),
            column_types={"id": "INTEGER"},
        )
