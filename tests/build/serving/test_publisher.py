"""Tests for build-side serving snapshot publishing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pytest

from codeintel.build.hamilton.native.export.serving_artifacts import (
    SERVING_ARTIFACT_BUILDSPEC,
    SERVING_ARTIFACT_DATASET_MANIFEST_PATHS,
    SERVING_ARTIFACT_SCHEMA_MANIFEST,
    SERVING_ARTIFACT_SEMANTIC_REGISTRY,
    SERVING_ARTIFACTS_TARGET_NAME,
)
from codeintel.core.schemas.hashing import schema_hash as compute_schema_hash
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.publisher import (
    PublishServingSnapshotRequest,
    publish_serving_snapshot,
)
from codeintel.storage.constants import META_CATALOG_NAME
from codeintel.storage.contracts.schema_provider import get_schema_provider
from codeintel.storage.datasets.arrow_store import ArrowDatasetWriteOptions, write_dataset
from codeintel.storage.datasets.manifests import dataset_manifest_path
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.metadata.meta_catalog import attach_meta_database, meta_table_ref
from tests._helpers.assertions import (
    assert_record_has_artifacts,
    assert_target_ok,
    expect_equal,
    expect_true,
)
from tests._helpers.harnesses.serving_harness import ServingTargetHarness
from tests._helpers.schemas import ensure_production_schemas, ensure_storage_contract_catalog

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.core.manifests import ServingSnapshotManifest


@dataclass(frozen=True)
class _StubGateway:
    config: StorageConfig
    con: duckdb.DuckDBPyConnection

    def execute(
        self, sql: str, params: Sequence[object] | None = None
    ) -> duckdb.DuckDBPyConnection:
        return self.con.execute(sql, params)


@dataclass(frozen=True)
class _ModulesDataset:
    modules_entry: dict[str, object]
    modules_hash: str
    dataset_manifest_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _PublisherSpecs:
    semantic_registry: Path
    schema_manifest: Path
    buildspec: Path


def _write_text(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_modules(con: duckdb.DuckDBPyConnection, *, repo: str, commit: str) -> None:
    ensure_production_schemas(con)
    con.execute(
        "INSERT INTO core.modules (repo, commit, module, path, row_hash) VALUES (?, ?, ?, ?, ?)",
        [repo, commit, "pkg.foo", "foo.py", "seed"],
    )


def _prepare_modules_dataset(
    tmp_path: Path,
    *,
    con: duckdb.DuckDBPyConnection,
    commit: str,
) -> _ModulesDataset:
    ensure_storage_contract_catalog()
    schema_provider = get_schema_provider()
    modules_schema = schema_provider.require_table_schema("core.modules")
    modules_hash = compute_schema_hash(modules_schema)
    modules_entry = modules_schema.to_json_obj()
    modules_entry["schema_hash"] = modules_hash

    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)
    arrow_table = con.execute("SELECT * FROM core.modules").fetch_arrow_table()
    write_dataset(
        dataset_root=dataset_root,
        table_key="core.modules",
        snapshot_id=commit,
        data=arrow_table,
        options=ArrowDatasetWriteOptions(schema_hash=modules_hash),
    )
    manifest_path = dataset_manifest_path(
        dataset_root=dataset_root,
        table_key="core.modules",
        snapshot_id=commit,
    )
    return _ModulesDataset(
        modules_entry=modules_entry,
        modules_hash=modules_hash,
        dataset_manifest_paths=(manifest_path,),
    )


def _write_publisher_specs(
    tmp_path: Path,
    *,
    modules_entry: dict[str, object],
    modules_hash: str,
) -> _PublisherSpecs:
    semantic_registry = tmp_path / "semantic_registry.json"
    schema_manifest = tmp_path / "schema_manifest.json"
    buildspec = tmp_path / "buildspec.json"
    _write_text(semantic_registry, {"version": "v1", "views": []})
    _write_text(
        schema_manifest,
        {"version": "v2", "tables": [modules_entry], "views": [], "artifacts": []},
    )
    _write_text(
        buildspec,
        {
            "spec_version": 1,
            "targets": [],
            "datasets": [{"table_key": "core.modules", "schema_hash": modules_hash}],
        },
    )
    return _PublisherSpecs(
        semantic_registry=semantic_registry,
        schema_manifest=schema_manifest,
        buildspec=buildspec,
    )


def _publish_snapshot(
    tmp_path: Path,
    *,
    repo: str,
    commit: str,
    run_id: str,
    keep_last: int,
) -> tuple[ServingSnapshotManifest, ServingSnapshotPointer, Path]:
    db_path = tmp_path / "build.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE t (id INTEGER)")
        con.execute("INSERT INTO t VALUES (1)")
        _seed_modules(con, repo=repo, commit=commit)
        modules_dataset = _prepare_modules_dataset(
            tmp_path,
            con=con,
            commit=commit,
        )

        gateway = _StubGateway(
            config=StorageConfig(db_path=db_path, repo=repo, commit=commit),
            con=con,
        )

        specs = _write_publisher_specs(
            tmp_path,
            modules_entry=modules_dataset.modules_entry,
            modules_hash=modules_dataset.modules_hash,
        )

        serve_dir = tmp_path / "serve"
        manifest = publish_serving_snapshot(
            gateway=gateway,
            request=PublishServingSnapshotRequest(
                run_id=run_id,
                serve_dir=serve_dir,
                semantic_registry_path=specs.semantic_registry,
                schema_manifest_path=specs.schema_manifest,
                buildspec_path=specs.buildspec,
                dataset_manifest_paths=modules_dataset.dataset_manifest_paths,
                keep_last=keep_last,
            ),
        )
    finally:
        con.close()

    pointer = ServingSnapshotPointer.load(tmp_path / "serve" / "current.json")
    return manifest, pointer, tmp_path / "serve"


def test_publish_serving_snapshot_creates_snapshot_and_pointer(tmp_path: Path) -> None:
    """Publisher checkpoints, copies artifacts, and updates current.json atomically."""
    manifest, pointer, serve_dir = _publish_snapshot(
        tmp_path,
        repo="demo/repo",
        commit="c1",
        run_id="run-1",
        keep_last=10,
    )
    snap_dir = serve_dir / "snapshots" / pointer.run_id
    expect_true((snap_dir / "codeintel.duckdb").exists())
    expect_true((snap_dir / "semantic_registry.json").exists())
    expect_true((snap_dir / "schema_manifest.json").exists())
    expect_true((snap_dir / "buildspec.json").exists())
    expect_true((snap_dir / "snapshot_manifest.json").exists())
    expect_true((serve_dir / "current.json").exists())

    expect_equal(pointer.run_id, "run-1")
    expect_equal(pointer.repo, "demo/repo")
    expect_equal(pointer.commit, "c1")
    expect_equal(pointer.snapshot_root, snap_dir)
    expect_equal(pointer.semantic_layer_version, manifest.semantic_layer_version)
    expect_true(pointer.buildspec_path.exists())
    expect_true(pointer.snapshot_manifest_path.exists())
    expect_true("core.modules" in manifest.datasets)

    snap_con = duckdb.connect(pointer.db_path, read_only=True)
    try:
        config = StorageConfig(
            db_path=Path(pointer.db_path),
            read_only=True,
            apply_schema=False,
            ensure_views=False,
            validate_schema=False,
        )
        attach_meta_database(snap_con, config=config)
        present = snap_con.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'docs' AND table_name = 'search_documents'
            LIMIT 1
            """
        ).fetchone()
        expect_true(present is not None)
        lineage_edges = snap_con.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_catalog = ?
              AND table_schema = 'metadata'
              AND table_name = 'derived_lineage_edges'
            LIMIT 1
            """,
            [META_CATALOG_NAME],
        ).fetchone()
        expect_true(lineage_edges is not None)
        lineage_columns = snap_con.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_catalog = ?
              AND table_schema = 'metadata'
              AND table_name = 'derived_lineage_columns'
            LIMIT 1
            """,
            [META_CATALOG_NAME],
        ).fetchone()
        expect_true(lineage_columns is not None)
    finally:
        snap_con.close()


def test_publish_serving_snapshot_retention(tmp_path: Path) -> None:
    """Retention keeps only the newest snapshots."""
    db_path = tmp_path / "build.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE t (id INTEGER)")
    _seed_modules(con, repo="demo/repo", commit="c1")
    modules_dataset = _prepare_modules_dataset(
        tmp_path,
        con=con,
        commit="c1",
    )

    gateway = _StubGateway(
        config=StorageConfig(db_path=db_path, repo="demo/repo", commit="c1"), con=con
    )

    specs = _write_publisher_specs(
        tmp_path,
        modules_entry=modules_dataset.modules_entry,
        modules_hash=modules_dataset.modules_hash,
    )

    serve_dir = tmp_path / "serve"
    publish_serving_snapshot(
        gateway=gateway,
        request=PublishServingSnapshotRequest(
            run_id="run-1",
            serve_dir=serve_dir,
            semantic_registry_path=specs.semantic_registry,
            schema_manifest_path=specs.schema_manifest,
            buildspec_path=specs.buildspec,
            dataset_manifest_paths=modules_dataset.dataset_manifest_paths,
            keep_last=1,
        ),
    )
    publish_serving_snapshot(
        gateway=gateway,
        request=PublishServingSnapshotRequest(
            run_id="run-2",
            serve_dir=serve_dir,
            semantic_registry_path=specs.semantic_registry,
            schema_manifest_path=specs.schema_manifest,
            buildspec_path=specs.buildspec,
            dataset_manifest_paths=modules_dataset.dataset_manifest_paths,
            keep_last=1,
        ),
    )
    con.close()

    snaps = sorted([p.name for p in (serve_dir / "snapshots").iterdir() if p.is_dir()])
    expect_equal(snaps, ["run-2"])


def test_publish_serving_snapshot_fails_on_empty_search_docs(tmp_path: Path) -> None:
    """Publisher fails when docs.search_documents is empty."""
    db_path = tmp_path / "build.duckdb"
    con = duckdb.connect(str(db_path))
    ensure_production_schemas(con)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS core.modules (
            repo VARCHAR,
            commit VARCHAR,
            module VARCHAR,
            path VARCHAR,
            row_hash VARCHAR
        )
        """
    )
    modules_dataset = _prepare_modules_dataset(
        tmp_path,
        con=con,
        commit="c1",
    )

    gateway = _StubGateway(
        config=StorageConfig(db_path=db_path, repo="demo/repo", commit="c1"), con=con
    )

    specs = _write_publisher_specs(
        tmp_path,
        modules_entry=modules_dataset.modules_entry,
        modules_hash=modules_dataset.modules_hash,
    )

    serve_dir = tmp_path / "serve"
    with pytest.raises(RuntimeError, match="Search index build failed"):
        publish_serving_snapshot(
            gateway=gateway,
            request=PublishServingSnapshotRequest(
                run_id="run-empty",
                serve_dir=serve_dir,
                semantic_registry_path=specs.semantic_registry,
                schema_manifest_path=specs.schema_manifest,
                buildspec_path=specs.buildspec,
                dataset_manifest_paths=modules_dataset.dataset_manifest_paths,
                keep_last=10,
            ),
        )
    con.close()


def test_publish_serving_snapshot_fails_on_missing_lineage_tables(tmp_path: Path) -> None:
    """Publisher fails when derived lineage tables are missing."""
    db_path = tmp_path / "build.duckdb"
    con = duckdb.connect(str(db_path))
    _seed_modules(con, repo="demo/repo", commit="c1")
    modules_dataset = _prepare_modules_dataset(
        tmp_path,
        con=con,
        commit="c1",
    )
    edges_ref = meta_table_ref("metadata.derived_lineage_edges")
    columns_ref = meta_table_ref("metadata.derived_lineage_columns")
    con.execute(f"DROP TABLE IF EXISTS {edges_ref}")
    con.execute(f"DROP TABLE IF EXISTS {columns_ref}")

    gateway = _StubGateway(
        config=StorageConfig(db_path=db_path, repo="demo/repo", commit="c1"), con=con
    )

    specs = _write_publisher_specs(
        tmp_path,
        modules_entry=modules_dataset.modules_entry,
        modules_hash=modules_dataset.modules_hash,
    )

    serve_dir = tmp_path / "serve"
    with pytest.raises(RuntimeError, match="Lineage metadata missing"):
        publish_serving_snapshot(
            gateway=gateway,
            request=PublishServingSnapshotRequest(
                run_id="run-no-lineage",
                serve_dir=serve_dir,
                semantic_registry_path=specs.semantic_registry,
                schema_manifest_path=specs.schema_manifest,
                buildspec_path=specs.buildspec,
                dataset_manifest_paths=modules_dataset.dataset_manifest_paths,
                keep_last=10,
            ),
        )
    con.close()


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
            SERVING_ARTIFACT_DATASET_MANIFEST_PATHS,
        ),
    )
    manifest = serving_target_harness.publish_snapshot(run_id="publisher-harness")
    expect_true(Path(manifest.db_path).is_file())
