"""Ensure semantic registry compilation uses driver tags."""

from __future__ import annotations

import json
import types

import hamilton.driver as h_driver

from codeintel.build.serving.semantic_compile import (
    compile_semantic_registry_from_driver,
)
from codeintel.core.hamilton.semantic_tags import semantic_view
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider


@semantic_view(
    semantic_id="sv_example",
    table_key="docs.v_example",
    entity="example",
    grain="example",
)
def semantic_view_example() -> int:
    return 1


def _driver() -> h_driver.Driver:
    module = types.ModuleType("semantic_registry_fixture")
    setattr(module, semantic_view_example.__name__, semantic_view_example)
    return h_driver.Builder().with_modules(module).build()


def test_compile_semantic_registry_from_driver_tags() -> None:
    provider = MappingSchemaProvider(
        {
            "docs.v_example": TableSchema(
                schema="docs",
                name="v_example",
                columns=[
                    Column(name="id", type="BIGINT"),
                    Column(name="name", type="VARCHAR"),
                ],
            )
        }
    )
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
