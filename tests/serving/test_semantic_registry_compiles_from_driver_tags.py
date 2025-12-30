"""Ensure semantic registry compilation uses driver tags."""

from __future__ import annotations

import json
import sys
import types

import hamilton.driver as h_driver
from hamilton.function_modifiers import tag as h_tag
from sqlglot import exp, parse_one

from codeintel.build.serving.semantic_compile import (
    compile_semantic_registry_from_driver,
)
from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.semantic_tags import semantic_view
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.storage.views.schema_inference import derive_view_schemas


@semantic_view(
    semantic_id="sv_example",
    table_key="docs.v_example",
    entity="example",
    grain="example",
)
def semantic_view_example() -> int:
    """Return sentinel value for semantic view example.

    Returns
    -------
    int
        Sentinel value for testing.
    """
    return 1


def _driver() -> h_driver.Driver:
    """Build a Driver with the semantic view fixture module.

    Returns
    -------
    h_driver.Driver
        Driver seeded with the semantic view example.
    """
    module = types.ModuleType("semantic_registry_fixture")
    module_name = module.__name__
    original_module = semantic_view_example.__module__
    semantic_view_example.__module__ = module.__name__
    try:
        sys.modules[module_name] = module
        setattr(module, semantic_view_example.__name__, semantic_view_example)
        return h_driver.Builder().with_modules(module).build()
    finally:
        sys.modules.pop(module_name, None)
        semantic_view_example.__module__ = original_module


def test_compile_semantic_registry_from_driver_tags() -> None:
    """Compile semantic registry using tags from a Driver."""
    provider = _schema_provider()
    compiled = compile_semantic_registry_from_driver(
        schema_provider=provider,
        dr=_driver(),
        version="v1",
    )
    payload = json.loads(compiled.to_json())
    assert payload["version"] == "v1"
    assert payload["views"][0]["id"] == "sv_example"
    assert payload["views"][0]["table_key"] == "docs.v_example"
    assert payload["views"][0]["columns"] == ["id", "name"]


def _schema_provider() -> MappingSchemaProvider:
    base_schema = TableSchema(
        schema="analytics",
        name="example",
        columns=[
            Column(name="id", type="BIGINT"),
            Column(name="name", type="VARCHAR"),
        ],
    )
    base_provider = MappingSchemaProvider({base_schema.table_key: base_schema})
    view_module = types.ModuleType("semantic_registry_schema_fixture")

    @h_tag(output_kind=ht.OUTPUT_KIND_VIEW, table_key="docs.v_example")
    def v_example() -> exp.Expression:
        return parse_one("SELECT id, name FROM analytics.example", read="duckdb")

    v_example.__module__ = view_module.__name__
    view_module.__dict__["v_example"] = v_example
    derived = derive_view_schemas(provider=base_provider, modules=(view_module,))
    return MappingSchemaProvider(derived)
