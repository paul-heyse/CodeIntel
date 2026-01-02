"""PR-85: semantic registry compilation is driven by Hamilton tag discovery."""

from __future__ import annotations

import hamilton.driver as h_driver

from codeintel.build.hamilton.native.views import view_outputs
from codeintel.core.hamilton.tag_query import TagQuery
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.serving.semantic.registry_compiler import compile_semantic_registry
from tests._helpers.assertions.expectation_assertions import expect_true


def test_semantic_registry_compiles_from_hamilton_tags() -> None:
    """Discover semantic tags via Hamilton and compile a deterministic registry."""
    driver = h_driver.Builder().with_modules(view_outputs).allow_module_overrides().build()
    provider = MappingSchemaProvider(
        schemas={
            "docs.v_function_architecture": TableSchema(
                schema="docs",
                name="v_function_architecture",
                columns=[
                    Column(name="repo", type="VARCHAR", nullable=False),
                    Column(name="commit", type="VARCHAR", nullable=False),
                    Column(name="qualname", type="VARCHAR", nullable=False),
                ],
            )
        }
    )

    compiled = compile_semantic_registry(
        schema_provider=provider,
        tag_query=TagQuery(driver),
    )
    expect_true(any(v.get("id") == "function.architecture" for v in compiled.views))
