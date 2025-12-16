"""Tests for semantic registry compilation from tags."""

from __future__ import annotations

from codeintel.build.hamilton import tags as ht
from codeintel.build.serving.semantic_compile import compile_semantic_registry_from_views
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.hamilton.semantic_tags import (
    TAG_MCP_VISIBLE,
    TAG_OUTPUT_KIND,
    TAG_SEMANTIC_COLS,
    TAG_SEMANTIC_ENTITY,
    TAG_SEMANTIC_GRAIN,
    TAG_SEMANTIC_ID,
    TAG_TABLE_KEY,
)
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_compile_registry_uses_explicit_columns_when_present() -> None:
    """Explicit semantic_columns overrides SchemaProvider-derived columns."""
    provider = MappingSchemaProvider(
        schemas={
            "docs.v_demo": TableSchema(
                schema="docs",
                name="v_demo",
                columns=[Column(name="id", type="INTEGER", nullable=False)],
            )
        }
    )
    tags = {
        "docs.v_demo": {
            TAG_OUTPUT_KIND: ht.OUTPUT_KIND_SEMANTIC_VIEW,
            TAG_MCP_VISIBLE: "1",
            TAG_SEMANTIC_ID: "demo.view",
            TAG_TABLE_KEY: "docs.v_demo",
            TAG_SEMANTIC_ENTITY: "demo",
            TAG_SEMANTIC_GRAIN: "per_row",
            TAG_SEMANTIC_COLS: "id, label",
        }
    }

    compiled = compile_semantic_registry_from_views(schema_provider=provider, view_tags=tags)
    expect_equal(compiled.views[0]["columns"], ["id", "label"])


def test_compile_registry_filters_non_semantic_and_hidden() -> None:
    """Non-semantic or mcp_visible=0 views are excluded."""
    provider = MappingSchemaProvider(schemas={})
    tags = {
        "one": {TAG_OUTPUT_KIND: ht.OUTPUT_KIND_SEMANTIC_VIEW, TAG_MCP_VISIBLE: "0"},
        "two": {TAG_OUTPUT_KIND: "other", TAG_MCP_VISIBLE: "1"},
    }

    compiled = compile_semantic_registry_from_views(schema_provider=provider, view_tags=tags)
    expect_equal(compiled.views, [])
