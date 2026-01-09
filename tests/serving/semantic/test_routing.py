"""Tests for semantic engine routing decisions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from sqlglot import exp, parse_one

from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.semantic.arrow_plan_builder import ArrowPlanSpec
from codeintel.serving.semantic.engines.protocol import EngineContext
from codeintel.serving.semantic.inventory import SchemaInventory
from codeintel.serving.semantic.models import SemanticViewSpec
from codeintel.serving.semantic.query_ast import ServingQuery
from codeintel.serving.semantic.registry import SemanticRegistry
from codeintel.serving.semantic.routing import auto_preference
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.serving.settings import ServingSettings
from codeintel.storage.constants import DUCKDB_DIALECT
from codeintel.storage.datasets.manifest_index import DatasetManifestIndex
from codeintel.storage.datasets.scanning import QueryPlanSpec

if TYPE_CHECKING:
    from pathlib import Path


def _engine_context(
    tmp_path: Path,
    *,
    view: SemanticViewSpec,
) -> EngineContext:
    pointer = ServingSnapshotPointer(
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
    registry = SemanticRegistry(version="v1", views=(view,))
    inventory = SchemaInventory(schemas={})
    dataset_manifests = DatasetManifestIndex(by_table_key={})
    settings = ServingSettings(serve_dir=tmp_path)
    return EngineContext(
        pointer=pointer,
        inventory=inventory,
        registry=registry,
        dataset_manifests=dataset_manifests,
        settings=settings,
        warehouse=None,
    )


def _serving_query(
    ast_sql: str,
    *,
    view: SemanticViewSpec,
    arrow_plan: ArrowPlanSpec | None = None,
) -> ServingQuery:
    spec = SemanticQuerySpec(
        view_id=view.id,
        table_key=view.table_key,
        allowed_columns=frozenset(view.columns),
        columns=list(view.columns),
        filters=[],
        order_by=[],
        limit=100,
        offset=0,
        column_types=None,
    )
    ast = cast("exp.Select", parse_one(ast_sql, dialect=DUCKDB_DIALECT))
    plan_spec = QueryPlanSpec(
        table_key=view.table_key,
        columns=tuple(view.columns),
        filter_expression=None,
    )
    return ServingQuery(
        spec=spec,
        ast=ast,
        plan_spec=plan_spec,
        arrow_plan=arrow_plan,
    )


def test_auto_preference_prefers_duckdb_for_unregistered_views(tmp_path: Path) -> None:
    """Views without engine-specific requirements should route to DuckDB."""
    view = SemanticViewSpec(
        id="demo.view",
        kind="view",
        table_key="docs.v_demo",
        entity="demo",
        grain="per_row",
        columns=["id"],
    )
    ctx = _engine_context(tmp_path, view=view)
    query = _serving_query("SELECT id FROM docs.v_demo", view=view)
    assert auto_preference(query, ctx=ctx) == ("duckdb",)


def test_auto_preference_prefers_polars_for_tables(tmp_path: Path) -> None:
    """Table-backed views should route to DuckDB in auto mode."""
    view = SemanticViewSpec(
        id="demo.table",
        kind="table",
        table_key="docs.v_demo",
        entity="demo",
        grain="per_row",
        columns=["id"],
    )
    ctx = _engine_context(tmp_path, view=view)
    query = _serving_query("SELECT id FROM docs.v_demo", view=view)
    assert auto_preference(query, ctx=ctx) == ("duckdb",)


def test_auto_preference_prefers_arrow_when_plan_present(tmp_path: Path) -> None:
    """Arrow plan hints should prefer the Arrow engine in auto mode."""
    view = SemanticViewSpec(
        id="demo.arrow",
        kind="view",
        table_key="docs.v_demo",
        entity="demo",
        grain="per_row",
        columns=["id"],
    )
    ctx = _engine_context(tmp_path, view=view)
    query = _serving_query(
        "SELECT id FROM docs.v_demo",
        view=view,
        arrow_plan=cast("ArrowPlanSpec", object()),
    )
    assert auto_preference(query, ctx=ctx) == ("arrow", "duckdb")
