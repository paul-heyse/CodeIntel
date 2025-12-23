"""Tests for build-side serving snapshot publishing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from codeintel.build.hamilton.native.export.serving_artifacts import (
    SERVING_ARTIFACT_BUILDSPEC,
    SERVING_ARTIFACT_SCHEMA_MANIFEST,
    SERVING_ARTIFACT_SEMANTIC_REGISTRY,
    SERVING_ARTIFACTS_TARGET_NAME,
)
from codeintel.build.serving.publisher import (
    PublishServingSnapshotRequest,
    publish_serving_snapshot,
)
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.storage.gateway.config import StorageConfig
from tests._helpers.assertions import (
    assert_record_has_artifacts,
    assert_target_ok,
    expect_equal,
    expect_true,
)
from tests._helpers.harnesses.serving_harness import ServingTargetHarness

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    buildspec = tmp_path / "buildspec.json"
    _write_text(semantic_registry, {"version": "v1", "views": []})
    _write_text(schema_manifest, {"version": "v1", "tables": []})
    _write_text(buildspec, {"spec_version": 1, "targets": [], "datasets": []})

    serve_dir = tmp_path / "serve"
    manifest = publish_serving_snapshot(
        gateway=gateway,
        request=PublishServingSnapshotRequest(
            run_id="run-1",
            serve_dir=serve_dir,
            semantic_registry_path=semantic_registry,
            schema_manifest_path=schema_manifest,
            buildspec_path=buildspec,
            keep_last=10,
        ),
    )
    con.close()

    snap_dir = serve_dir / "snapshots" / "run-1"
    expect_true((snap_dir / "codeintel.duckdb").exists())
    expect_true((snap_dir / "semantic_registry.json").exists())
    expect_true((snap_dir / "schema_manifest.json").exists())
    expect_true((snap_dir / "buildspec.json").exists())
    expect_true((serve_dir / "current.json").exists())

    pointer = ServingSnapshotPointer.load(serve_dir / "current.json")
    expect_equal(pointer.run_id, "run-1")
    expect_equal(pointer.repo, "demo/repo")
    expect_equal(pointer.commit, "c1")
    expect_equal(pointer.semantic_layer_version, manifest.semantic_layer_version)
    expect_true(pointer.buildspec_path.exists())

    snap_con = duckdb.connect(pointer.db_path, read_only=True)
    try:
        present = snap_con.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'docs' AND table_name = 'search_documents'
            LIMIT 1
            """
        ).fetchone()
        expect_true(present is not None)
    finally:
        snap_con.close()


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
    buildspec = tmp_path / "buildspec.json"
    _write_text(semantic_registry, {"version": "v1", "views": []})
    _write_text(schema_manifest, {"version": "v1", "tables": []})
    _write_text(buildspec, {"spec_version": 1, "targets": [], "datasets": []})

    serve_dir = tmp_path / "serve"
    publish_serving_snapshot(
        gateway=gateway,
        request=PublishServingSnapshotRequest(
            run_id="run-1",
            serve_dir=serve_dir,
            semantic_registry_path=semantic_registry,
            schema_manifest_path=schema_manifest,
            buildspec_path=buildspec,
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
            buildspec_path=buildspec,
            keep_last=1,
        ),
    )
    con.close()

    snaps = sorted([p.name for p in (serve_dir / "snapshots").iterdir() if p.is_dir()])
    expect_equal(snaps, ["run-2"])


def test_serving_harness_publishes_snapshot(
    serving_target_harness: ServingTargetHarness,
) -> None:
    """Run serving_artifacts and publish a snapshot via the harness."""
    records = serving_target_harness.run_targets([SERVING_ARTIFACTS_TARGET_NAME])
    record = records[SERVING_ARTIFACTS_TARGET_NAME]
    assert_target_ok(record)
    assert_record_has_artifacts(
        record,
        (
            SERVING_ARTIFACT_SEMANTIC_REGISTRY,
            SERVING_ARTIFACT_SCHEMA_MANIFEST,
            SERVING_ARTIFACT_BUILDSPEC,
        ),
    )
    manifest = serving_target_harness.publish_snapshot(run_id="publisher-harness")
    expect_true(Path(manifest.db_path).is_file())
