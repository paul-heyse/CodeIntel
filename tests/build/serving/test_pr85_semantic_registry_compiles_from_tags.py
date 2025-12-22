"""PR-85: semantic registry compilation is driven by Hamilton tag discovery."""

from __future__ import annotations

from codeintel.build.hamilton.tag_index import TagIndex
from codeintel.build.serving.semantic_compile import compile_semantic_registry_from_views
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.storage.views import ibis_views
from tests._helpers.assertions.expectation_assertions import expect_true


def test_semantic_registry_compiles_from_hamilton_tags() -> None:
    """Discover semantic tags via Hamilton and compile a deterministic registry."""
    tags = TagIndex.from_modules(modules=(ibis_views,)).semantic_view_tags()
    expect_true("docs.v_function_summary" in tags)

    provider = MappingSchemaProvider(
        schemas={
            "docs.v_function_summary": TableSchema(
                schema="docs",
                name="v_function_summary",
                columns=[
                    Column(name="repo", type="VARCHAR", nullable=False),
                    Column(name="commit", type="VARCHAR", nullable=False),
                    Column(name="qualname", type="VARCHAR", nullable=False),
                ],
            )
        }
    )

    compiled = compile_semantic_registry_from_views(schema_provider=provider, view_tags=tags)
    expect_true(any(v.get("id") == "function.summary" for v in compiled.views))
