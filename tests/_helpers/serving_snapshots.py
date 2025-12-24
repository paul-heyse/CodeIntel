"""Helpers for constructing serving snapshot fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb

from codeintel.config.primitives import BuildPaths
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.storage.serving.search_index import build_search_documents_table
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT
from tests._helpers.hamilton_harness_artifacts import HarnessArtifacts
from tests._helpers.schemas import ensure_production_schemas

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class DemoSnapshotPaths:
    pointer_path: Path
    db_path: Path
    registry_path: Path
    manifest_path: Path
    buildspec_path: Path


def _write_demo_db(db_path: Path, *, row_count: int) -> None:
    con = duckdb.connect(str(db_path))
    ensure_production_schemas(con)
    con.execute("CREATE TABLE docs.v_demo (id INTEGER, label VARCHAR)")
    con.execute(
        "INSERT INTO docs.v_demo SELECT i, 'label-' || i::VARCHAR FROM range(1, ?) t(i)",
        [row_count + 1],
    )
    build_search_documents_table(con)
    con.close()


def _write_registry(path: Path) -> None:
    artifacts = HarnessArtifacts(
        repo_root=path.parent,
        paths=BuildPaths.from_explicit(build_dir=path.parent),
    )
    artifacts.write_semantic_registry(
        path=path,
        views=[
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
            }
        ],
    )


def _write_schema_manifest(path: Path) -> None:
    artifacts = HarnessArtifacts(
        repo_root=path.parent,
        paths=BuildPaths.from_explicit(build_dir=path.parent),
    )
    artifacts.write_schema_manifest(
        path=path,
        tables=[
            {
                "schema": "docs",
                "name": "v_demo",
                "table_key": "docs.v_demo",
                "primary_key": ["id"],
                "indexes": [],
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "label", "type": "VARCHAR", "nullable": True},
                ],
            }
        ],
    )


def _write_buildspec(path: Path) -> None:
    artifacts = HarnessArtifacts(
        repo_root=path.parent,
        paths=BuildPaths.from_explicit(build_dir=path.parent),
    )
    artifacts.write_buildspec(
        path=path,
        datasets=[{"table_key": "docs.v_demo", "schema_hash": "schema_v_demo"}],
    )


def _write_pointer(
    path: Path,
    *,
    db_path: Path,
    registry_path: Path,
    manifest_path: Path,
    buildspec_path: Path,
) -> None:
    pointer = ServingSnapshotPointer(
        db_path=db_path,
        semantic_registry_path=registry_path,
        schema_manifest_path=manifest_path,
        buildspec_path=buildspec_path,
        repo=DEFAULT_VARIANT.repo,
        commit=DEFAULT_VARIANT.commit,
        run_id="run-1",
        published_at=datetime.now(tz=UTC),
        semantic_layer_version="v123",
    )
    path.write_text(pointer.to_json(), encoding="utf-8")


def setup_demo_snapshot(
    root: Path,
    *,
    pointer_path: Path | None = None,
    row_count: int = 3,
) -> DemoSnapshotPaths:
    """Create a minimal serving snapshot with required metadata tables.

    Returns
    -------
    DemoSnapshotPaths
        Paths to the snapshot database, artifacts, and pointer file.
    """
    db_path = root / "codeintel.duckdb"
    registry_path = root / "semantic_registry.json"
    manifest_path = root / "schema_manifest.json"
    buildspec_path = root / "buildspec.json"
    pointer_path = pointer_path or root / "current.json"

    _write_demo_db(db_path, row_count=row_count)
    _write_registry(registry_path)
    _write_schema_manifest(manifest_path)
    _write_buildspec(buildspec_path)
    _write_pointer(
        pointer_path,
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )
    return DemoSnapshotPaths(
        pointer_path=pointer_path,
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )


__all__ = ["DemoSnapshotPaths", "setup_demo_snapshot"]
