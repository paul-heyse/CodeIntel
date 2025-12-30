"""PR-85: semantic registry compilation is driven by Hamilton tag discovery."""

from __future__ import annotations

import hamilton.driver as h_driver

from codeintel.build.serving.semantic_compile import compile_semantic_registry_from_driver
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.core.schemas.table_registry import TABLE_SCHEMAS
from codeintel.storage.views import sqlglot_views
from codeintel.storage.views.schema_inference import derive_view_schemas
from tests._helpers.assertions.expectation_assertions import expect_true


def test_semantic_registry_compiles_from_hamilton_tags() -> None:
    """Discover semantic tags via Hamilton and compile a deterministic registry."""
    driver = h_driver.Builder().with_modules(sqlglot_views).allow_module_overrides().build()
    base_provider = MappingSchemaProvider(TABLE_SCHEMAS)
    derived = derive_view_schemas(provider=base_provider, modules=(sqlglot_views,))
    provider = MappingSchemaProvider({**TABLE_SCHEMAS, **derived})

    compiled = compile_semantic_registry_from_driver(schema_provider=provider, dr=driver)
    expect_true(any(v.get("id") == "function.summary" for v in compiled.views))
