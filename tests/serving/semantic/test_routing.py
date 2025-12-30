"""Tests for semantic engine routing decisions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from sqlglot import exp, parse_one

from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.semantic.engines.protocol import EngineContext
from codeintel.serving.semantic.inventory import SchemaInventory
from codeintel.serving.semantic.models import SemanticViewSpec
from codeintel.serving.semantic.query_ast import ServingQuery
from codeintel.serving.semantic.registry import SemanticRegistry
from codeintel.serving.semantic.routing import ast_supports_polars, auto_preference
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.serving.semantic.view_registry import ViewRegistry
from codeintel.serving.settings import ServingSettings
from codeintel.storage.constants import DUCKDB_DIALECT

if TYPE_CHECKING:
    from pathlib import Path


def _engine_context(
    tmp_path: Path,
    *,
    view: SemanticViewSpec,
    view_registry: ViewRegistry,
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
    settings = ServingSettings(serve_dir=tmp_path)
    return EngineContext(
        pointer=pointer,
        inventory=inventory,
        registry=registry,
        view_registry=view_registry,
        settings=settings,
        warehouse=None,
    )


def _serving_query(ast_sql: str, *, view: SemanticViewSpec) -> ServingQuery:
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
    return ServingQuery(spec=spec, ast=ast)


def test_ast_supports_polars_simple_select() -> None:
    """Simple selects should remain in the Polars envelope."""
    ast = parse_one("SELECT id FROM docs.v_demo", dialect=DUCKDB_DIALECT)
    assert ast_supports_polars(ast)


def test_ast_supports_polars_rejects_join() -> None:
    """Join nodes should route away from Polars."""
    ast = parse_one(
        "SELECT * FROM docs.v_demo d JOIN docs.v_demo b ON d.id = b.id",
        dialect=DUCKDB_DIALECT,
    )
    assert not ast_supports_polars(ast)


def test_ast_supports_polars_rejects_unknown_function() -> None:
    """Unsupported functions should route away from Polars."""
    ast = parse_one("SELECT foo(id) FROM docs.v_demo", dialect=DUCKDB_DIALECT)
    assert not ast_supports_polars(ast)


def test_auto_preference_prefers_duckdb_for_unregistered_views(tmp_path: Path) -> None:
    """Views without a registered Polars spec should route to DuckDB first."""
    view = SemanticViewSpec(
        id="demo.view",
        kind="view",
        table_key="docs.v_demo",
        entity="demo",
        grain="per_row",
        columns=["id"],
    )
    ctx = _engine_context(tmp_path, view=view, view_registry=ViewRegistry(specs={}))
    query = _serving_query("SELECT id FROM docs.v_demo", view=view)
    assert auto_preference(query, ctx=ctx) == ("duckdb", "polars")


def test_auto_preference_prefers_polars_for_tables(tmp_path: Path) -> None:
    """Table-backed views should prefer Polars when AST is compatible."""
    view = SemanticViewSpec(
        id="demo.table",
        kind="table",
        table_key="docs.v_demo",
        entity="demo",
        grain="per_row",
        columns=["id"],
    )
    ctx = _engine_context(tmp_path, view=view, view_registry=ViewRegistry(specs={}))
    query = _serving_query("SELECT id FROM docs.v_demo", view=view)
    assert auto_preference(query, ctx=ctx) == ("polars", "duckdb")
