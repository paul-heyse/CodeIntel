"""Tests for SemanticRegistry loading and lookup."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from codeintel.serving.semantic.registry import SemanticRegistry
from tests._helpers.assertions.expectation_assertions import expect_equal

if TYPE_CHECKING:
    from pathlib import Path


def test_registry_load_lookup_and_serialize(tmp_path: Path) -> None:
    """Registry loads views, supports lookup, and serializes deterministically."""
    registry_payload = {
        "version": "v1",
        "views": [
            {
                "id": "demo.view",
                "kind": "view",
                "table_key": "docs.v_demo",
                "entity": "demo",
                "grain": "per_row",
                "description": "Demo view",
                "primary_key": ["id"],
                "columns": ["id", "label"],
                "joins": [],
                "defaults": {"limit": 200, "order_by": ["id"]},
                "sensitivity": "internal",
                "deprecated": False,
                "replaced_by": None,
            }
        ],
    }
    path = tmp_path / "semantic_registry.json"
    path.write_text(json.dumps(registry_payload), encoding="utf-8")

    reg = SemanticRegistry.load(path)
    expect_equal(reg.version, "v1")
    expect_equal(reg.by_id("demo.view").table_key, "docs.v_demo")
    expect_equal(reg.list_view_ids(), ["demo.view"])

    encoded = reg.to_json()
    decoded = json.loads(encoded)
    expect_equal(decoded["version"], "v1")


def test_registry_unknown_view_raises(tmp_path: Path) -> None:
    """Unknown IDs raise KeyError."""
    path = tmp_path / "semantic_registry.json"
    path.write_text(json.dumps({"version": "v1", "views": []}), encoding="utf-8")
    reg = SemanticRegistry.load(path)

    with pytest.raises(KeyError, match="Unknown semantic view"):
        reg.by_id("missing")
