"""Tests for semantic planner behavior with derived view schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

from hamilton.function_modifiers import tag as h_tag
from sqlglot import exp, parse_one

from codeintel.build.spec import BuildSpec
from codeintel.core.hamilton import tags as ht
from codeintel.serving.db.manager import ServingSnapshotContext
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.semantic.inventory import SchemaInventory
from codeintel.serving.semantic.models import SemanticViewSpec
from codeintel.serving.semantic.planner import SemanticQueryPlanner
from codeintel.serving.semantic.registry import SemanticRegistry
from codeintel.serving.semantic.view_registry import ViewRegistry
from codeintel.serving.settings import ServingSettings

if TYPE_CHECKING:
    from codeintel.serving.db.manager import ServingDBManager


def test_planner_allows_columns_from_derived_views(tmp_path: Path) -> None:
    """Planner resolves allowed columns from a derived view schema."""
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

    inventory = SchemaInventory.load(path).with_derived_views(modules=(_view_module(),))
    view = SemanticViewSpec(
        id="demo.view",
        kind="view",
        table_key="docs.v_demo",
        entity="demo",
        grain="per_row",
        columns_dynamic=True,
    )
    registry = SemanticRegistry(version="v1", views=(view,))
    pointer = _pointer(tmp_path)
    context = ServingSnapshotContext(
        pointer=pointer,
        registry=registry,
        inventory=inventory,
        buildspec=BuildSpec(spec_version=1),
        view_registry=ViewRegistry.load(modules=()),
    )
    planner = SemanticQueryPlanner(
        db=cast("ServingDBManager", _StubDB(context)),
        settings=ServingSettings(serve_dir=tmp_path),
    )
    resolved = planner.resolve_view_context(pointer=pointer, view_id="demo.view")
    assert resolved.allowed_columns == ["id", "label"]


def _view_module() -> ModuleType:
    module = ModuleType("tests.semantic_planner_views")

    @h_tag(output_kind=ht.OUTPUT_KIND_VIEW, table_key="docs.v_demo")
    def v_demo() -> exp.Expression:
        return parse_one("SELECT id, label FROM analytics.demo", read="duckdb")

    v_demo.__module__ = module.__name__
    module.__dict__["v_demo"] = v_demo
    return module


def _pointer(tmp_path: Path) -> ServingSnapshotPointer:
    return ServingSnapshotPointer(
        snapshot_root=tmp_path,
        snapshot_manifest_path=tmp_path / "snapshot_manifest.json",
        db_path=tmp_path / "snapshot.duckdb",
        semantic_registry_path=tmp_path / "semantic_registry.json",
        schema_manifest_path=tmp_path / "schema_manifest.json",
        buildspec_path=tmp_path / "buildspec.json",
        repo="demo/repo",
        commit="deadbeef",
        run_id="run-1",
        published_at=datetime.now(tz=UTC),
        semantic_layer_version="v1",
    )


@dataclass(frozen=True)
class _StubDB:
    context: ServingSnapshotContext

    def snapshot_context(self, _pointer: ServingSnapshotPointer) -> ServingSnapshotContext:
        return self.context
