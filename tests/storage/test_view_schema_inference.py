"""Tests for SQLGlot view schema inference."""

from __future__ import annotations

from types import ModuleType

import pytest
from hamilton.function_modifiers import tag as h_tag
from sqlglot import exp, parse_one

from codeintel.core.hamilton import tags as ht
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.storage.views.schema_inference import derive_view_schemas

pytestmark = pytest.mark.no_runtime_env


def _view_module() -> ModuleType:
    module = ModuleType("tests.view_schema_module")

    @h_tag(output_kind=ht.OUTPUT_KIND_VIEW, table_key="docs.v_demo")
    def v_demo() -> exp.Expression:
        return parse_one("SELECT * FROM analytics.demo", read="duckdb")

    v_demo.__module__ = module.__name__
    module.__dict__["v_demo"] = v_demo
    return module


def test_derive_view_schema_from_sqlglot() -> None:
    """Derive a view schema using SQLGlot type annotation."""
    base_schema = TableSchema(
        schema="analytics",
        name="demo",
        columns=[
            Column(name="id", type="BIGINT", nullable=False),
            Column(name="name", type="VARCHAR", nullable=True),
        ],
    )
    provider = MappingSchemaProvider({base_schema.table_key: base_schema})
    view_module = _view_module()

    derived = derive_view_schemas(provider=provider, modules=(view_module,))
    view_schema = derived.get("docs.v_demo")
    if view_schema is None:
        pytest.fail("Expected docs.v_demo view schema to be derived")

    actual = [(col.name, col.type) for col in view_schema.columns]
    expected = [("id", "BIGINT"), ("name", "VARCHAR")]
    if actual != expected:
        pytest.fail(f"View schema mismatch: {actual} != {expected}")
