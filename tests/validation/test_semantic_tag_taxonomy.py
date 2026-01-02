"""Tests for semantic tag taxonomy validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.hamilton.validate import validate_nodes
from codeintel.core.hamilton import tags as ht

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class DummyNode:
    """Minimal node stub for semantic tag validation."""

    name: str
    tags: dict[str, object]
    dependencies: Sequence[DummyNode] = ()


def test_semantic_tag_taxonomy_requires_layer_and_version() -> None:
    """Require semantic layer and version tags on semantic outputs."""
    node = DummyNode(
        name="semantic_view_node",
        tags={
            ht.TAG_OUTPUT_KIND: ht.OUTPUT_KIND_SEMANTIC_VIEW,
            ht.TAG_SEMANTIC_ID: "function.architecture",
            ht.TAG_KIND: "table",
            ht.TAG_ENTITY: "function",
            ht.TAG_GRAIN: "per_function",
            ht.TAG_SCHEMA_REF: "semantic.function_architecture",
            ht.TAG_ENTITY_KEYS: "repo,commit,goid_h128",
            ht.TAG_JOIN_KEYS: "repo,commit,goid_h128",
            ht.TAG_TABLE_KEY: "docs.v_function_architecture",
        },
    )
    result = validate_nodes({node.name: node}, validate_schema=False)
    messages = [issue.message for issue in result.errors if issue.code == "missing_semantic_tag"]
    assert any("layer" in message for message in messages)
    assert any("version" in message for message in messages)
