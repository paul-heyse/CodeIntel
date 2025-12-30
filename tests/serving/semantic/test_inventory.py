"""Tests for SchemaInventory loading and access."""

from __future__ import annotations

import json
from types import ModuleType
from typing import TYPE_CHECKING

from hamilton.function_modifiers import tag as h_tag
from sqlglot import exp, parse_one

from codeintel.core.hamilton import tags as ht
from codeintel.serving.semantic.inventory import SchemaInventory
from tests._helpers.assertions.expectation_assertions import expect_equal

if TYPE_CHECKING:
    from pathlib import Path


def test_inventory_load_and_lookup(tmp_path: Path) -> None:
    """Load a schema manifest and access schemas by table_key."""
    manifest = {
        "version": "v2",
        "tables": [],
        "views": [
            {
                "schema": "docs",
                "name": "v_demo",
                "table_key": "docs.v_demo",
                "description": "demo view",
                "primary_key": ["id"],
                "indexes": [],
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "label", "type": "VARCHAR", "nullable": True},
                ],
            }
        ],
    }
    path = tmp_path / "schema_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    inv = SchemaInventory.load(path)
    schema = inv.require("docs.v_demo")
    expect_equal(schema.table_key, "docs.v_demo")
    expect_equal(schema.column_names(), ["id", "label"])


def test_inventory_summary_counts_docs_views(tmp_path: Path) -> None:
    """Summary differentiates docs.v_* views from other tables."""
    manifest = {
        "version": "v2",
        "tables": [
            {
                "schema": "analytics",
                "name": "function_metrics",
                "table_key": "analytics.function_metrics",
                "primary_key": [],
                "indexes": [],
                "columns": [{"name": "id", "type": "INTEGER", "nullable": False}],
            }
        ],
        "views": [
            {
                "schema": "docs",
                "name": "v_demo",
                "table_key": "docs.v_demo",
                "primary_key": [],
                "indexes": [],
                "columns": [{"name": "id", "type": "INTEGER", "nullable": False}],
            }
        ],
    }
    path = tmp_path / "schema_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    inv = SchemaInventory.load(path)
    expect_equal(inv.summary(), {"tables": 1, "views": 1})


def test_inventory_derives_docs_views(tmp_path: Path) -> None:
    """Derived docs views are merged into the inventory when missing."""
    manifest = {
        "version": "v2",
        "tables": [
            {
                "schema": "analytics",
                "name": "demo",
                "table_key": "analytics.demo",
                "primary_key": [],
                "indexes": [],
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "label", "type": "VARCHAR", "nullable": True},
                ],
            }
        ],
        "views": [],
    }
    path = tmp_path / "schema_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    inv = SchemaInventory.load(path).with_derived_views(modules=(_view_module(),))
    schema = inv.require("docs.v_demo")
    expect_equal(schema.table_key, "docs.v_demo")
    expect_equal(schema.column_names(), ["id", "label"])


def _view_module() -> ModuleType:
    module = ModuleType("tests.inventory_demo_views")

    @h_tag(output_kind=ht.OUTPUT_KIND_VIEW, table_key="docs.v_demo")
    def v_demo() -> exp.Expression:
        return parse_one("SELECT id, label FROM analytics.demo", read="duckdb")

    v_demo.__module__ = module.__name__
    module.__dict__["v_demo"] = v_demo
    return module
