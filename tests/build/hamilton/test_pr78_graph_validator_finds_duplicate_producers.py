"""Tests for PR-78: graph validator catches duplicate table producers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codeintel.build.hamilton.validate import validate_nodes
from codeintel.core.hamilton.tags import (
    NODE_TYPE_DATASET,
    NODE_TYPE_MATERIALIZE,
    TAG_DOMAIN,
    TAG_NODE_TYPE,
    TAG_TABLE_KEY,
    TAG_TARGET,
)


@dataclass(frozen=True)
class FakeNode:
    """Minimal node object for validator tests."""

    name: str
    tags: dict[str, object]
    dependencies: tuple[FakeNode, ...]


def test_pr78_graph_validator_finds_duplicate_producers() -> None:
    """Ensure validator detects two targets producing the same table_key."""
    table_key = "analytics.function_types"

    materialize_a = FakeNode(
        name="t__target_a",
        tags={
            TAG_NODE_TYPE: NODE_TYPE_MATERIALIZE,
            TAG_DOMAIN: "analytics",
            TAG_TARGET: "target_a",
        },
        dependencies=(),
    )
    materialize_b = FakeNode(
        name="t__target_b",
        tags={
            TAG_NODE_TYPE: NODE_TYPE_MATERIALIZE,
            TAG_DOMAIN: "analytics",
            TAG_TARGET: "target_b",
        },
        dependencies=(),
    )

    dataset_from_a = FakeNode(
        name="d__analytics__function_types__a",
        tags={
            TAG_NODE_TYPE: NODE_TYPE_DATASET,
            TAG_DOMAIN: "analytics",
            TAG_TABLE_KEY: table_key,
        },
        dependencies=(materialize_a,),
    )
    dataset_from_b = FakeNode(
        name="d__analytics__function_types__b",
        tags={
            TAG_NODE_TYPE: NODE_TYPE_DATASET,
            TAG_DOMAIN: "analytics",
            TAG_TABLE_KEY: table_key,
        },
        dependencies=(materialize_b,),
    )

    nodes = {
        materialize_a.name: materialize_a,
        materialize_b.name: materialize_b,
        dataset_from_a.name: dataset_from_a,
        dataset_from_b.name: dataset_from_b,
    }

    result = validate_nodes(nodes)
    codes = {e.code for e in result.errors}
    if "duplicate_table_key" not in codes:
        pytest.fail(f"Expected duplicate_table_key error, got: {sorted(codes)}")
