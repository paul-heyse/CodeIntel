"""Tests for build-side serving snapshot publishing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb

from codeintel.build.serving.publisher import (
    PublishServingSnapshotRequest,
    publish_serving_snapshot,
)
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.storage.gateway.config import StorageConfig
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class _StubGateway:
    config: StorageConfig
    con: duckdb.DuckDBPyConnection

    def execute(
        self, sql: str, params: Sequence[object] | None = None
    ) -> duckdb.DuckDBPyConnection:
        return self.con.execute(sql, params)


def _write_text(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_publish_serving_snapshot_creates_snapshot_and_pointer(tmp_path: Path) -> None:
    """Publisher checkpoints, copies artifacts, and updates current.json atomically."""
    db_path = tmp_path / "build.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE t (id INTEGER)")
    con.execute("INSERT INTO t VALUES (1)")

    gateway = _StubGateway(
        config=StorageConfig(db_path=db_path, repo="demo/repo", commit="c1"), con=con
    )

    semantic_registry = tmp_path / "semantic_registry.json"
    schema_manifest = tmp_path / "schema_manifest.json"
    _write_text(semantic_registry, {"version": "v1", "views": []})
    _write_text(schema_manifest, {"version": "v1", "tables": []})

    serve_dir = tmp_path / "serve"
    manifest = publish_serving_snapshot(
        gateway=gateway,
        request=PublishServingSnapshotRequest(
            run_id="run-1",
            serve_dir=serve_dir,
            semantic_registry_path=semantic_registry,
            schema_manifest_path=schema_manifest,
            keep_last=10,
        ),
    )
    con.close()

    snap_dir = serve_dir / "snapshots" / "run-1"
    expect_true((snap_dir / "codeintel.duckdb").exists())
    expect_true((snap_dir / "semantic_registry.json").exists())
    expect_true((snap_dir / "schema_manifest.json").exists())
    expect_true((serve_dir / "current.json").exists())

    pointer = ServingSnapshotPointer.load(serve_dir / "current.json")
    expect_equal(pointer.run_id, "run-1")
    expect_equal(pointer.repo, "demo/repo")
    expect_equal(pointer.commit, "c1")
    expect_equal(pointer.semantic_layer_version, manifest.semantic_layer_version)


def test_publish_serving_snapshot_retention(tmp_path: Path) -> None:
    """Retention keeps only the newest snapshots."""
    db_path = tmp_path / "build.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE t (id INTEGER)")

    gateway = _StubGateway(
        config=StorageConfig(db_path=db_path, repo="demo/repo", commit="c1"), con=con
    )

    semantic_registry = tmp_path / "semantic_registry.json"
    schema_manifest = tmp_path / "schema_manifest.json"
    _write_text(semantic_registry, {"version": "v1", "views": []})
    _write_text(schema_manifest, {"version": "v1", "tables": []})

    serve_dir = tmp_path / "serve"
    publish_serving_snapshot(
        gateway=gateway,
        request=PublishServingSnapshotRequest(
            run_id="run-1",
            serve_dir=serve_dir,
            semantic_registry_path=semantic_registry,
            schema_manifest_path=schema_manifest,
            keep_last=1,
        ),
    )
    publish_serving_snapshot(
        gateway=gateway,
        request=PublishServingSnapshotRequest(
            run_id="run-2",
            serve_dir=serve_dir,
            semantic_registry_path=semantic_registry,
            schema_manifest_path=schema_manifest,
            keep_last=1,
        ),
    )
    con.close()

    snaps = sorted([p.name for p in (serve_dir / "snapshots").iterdir() if p.is_dir()])
    expect_equal(snaps, ["run-2"])
