"""Serving snapshot publisher.

Publishes immutable read-only snapshots from the build database with atomic
pointer updates for zero-downtime deployments.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Protocol

from codeintel.core.manifests import ServingSnapshotManifest, SnapshotDatasetEntry
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.storage.constants import META_CATALOG_NAME, META_DB_FILENAME
from codeintel.storage.datasets.manifests import read_dataset_manifest
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.metadata.meta_catalog import resolve_meta_db_path
from codeintel.storage.serving.snapshot_service import (
    DatasetManifestError,
    LineageMetadataError,
    SearchIndexBuildError,
    ServingSnapshotService,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.storage.gateway.protocol import DuckDBConnection

log = logging.getLogger(__name__)


class SnapshotPublisherGateway(Protocol):
    """Protocol for the minimal gateway interface used by the snapshot publisher."""

    @property
    def config(self) -> StorageConfig: ...

    @property
    def con(self) -> DuckDBConnection: ...

    def execute(self, sql: str, params: Sequence[object] | None = None) -> DuckDBConnection: ...


@dataclass(frozen=True, slots=True)
class PublishServingSnapshotRequest:
    """Request payload for publishing a serving snapshot.

    Parameters
    ----------
    run_id
        Unique build run identifier.
    serve_dir
        Root serving directory containing `current.json` and `snapshots/`.
    semantic_registry_path
        Path to compiled semantic registry artifact.
    schema_manifest_path
        Path to compiled schema manifest artifact.
    dataset_manifest_paths
        Paths to dataset manifest artifacts (Arrow datasets).
    buildspec_path
        Path to compiled BuildSpec artifact.
    keep_last
        Number of old snapshots to retain.
    """

    run_id: str
    serve_dir: Path
    semantic_registry_path: Path
    schema_manifest_path: Path
    buildspec_path: Path
    dataset_manifest_paths: tuple[Path, ...] = ()
    keep_last: int = 10


@dataclass(frozen=True, slots=True)
class _SnapshotArtifacts:
    registry_path: Path
    schema_manifest_path: Path
    buildspec_path: Path


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically using rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(path.parent)) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    Path(tmp_path).replace(path)


def _compute_semantic_version(
    registry_path: Path,
    manifest_path: Path,
    buildspec_path: Path,
) -> str:
    """Compute semantic layer version hash.

    Returns
    -------
    str
        Short stable hash derived from registry, manifest, and buildspec bytes.
    """
    hasher = hashlib.sha256()
    for p in (registry_path, manifest_path, buildspec_path):
        if p.exists():
            hasher.update(p.read_bytes())
    return hasher.hexdigest()[:16]


def _prepare_snapshot_tables(
    *,
    snap_db: Path,
    run_id: str,
    snapshot_manifest: ServingSnapshotManifest,
) -> None:
    service = ServingSnapshotService()
    try:
        service.prepare_snapshot(db_path=snap_db, snapshot_manifest=snapshot_manifest)
    except SearchIndexBuildError as exc:
        log.exception(
            "build.serving.publisher.search_index_failed run_id=%s",
            run_id,
        )
        message = f"Search index build failed for serving snapshot run_id={run_id}"
        raise RuntimeError(message) from exc
    except LineageMetadataError as exc:
        log.exception(
            "build.serving.publisher.lineage_missing run_id=%s",
            run_id,
        )
        message = f"Lineage metadata missing for serving snapshot run_id={run_id}"
        raise RuntimeError(message) from exc
    except DatasetManifestError as exc:
        log.exception(
            "build.serving.publisher.dataset_manifest_failed run_id=%s",
            run_id,
        )
        message = f"Dataset manifest validation failed for serving snapshot run_id={run_id}"
        raise RuntimeError(message) from exc


def _copy_snapshot_database(*, snap_dir: Path, config: StorageConfig) -> Path:
    snap_db = snap_dir / "codeintel.duckdb"
    if config.attach_meta:
        meta_path = resolve_meta_db_path(config)
        if str(meta_path) != ":memory:" and meta_path.exists():
            snap_meta_db = snap_dir / META_DB_FILENAME
            shutil.copy2(meta_path, snap_meta_db)
    return snap_db


def _copy_snapshot_artifacts(
    request: PublishServingSnapshotRequest,
    *,
    snap_dir: Path,
) -> _SnapshotArtifacts:
    snap_registry = snap_dir / "semantic_registry.json"
    shutil.copy2(request.semantic_registry_path, snap_registry)

    snap_manifest = snap_dir / "schema_manifest.json"
    shutil.copy2(request.schema_manifest_path, snap_manifest)

    snap_buildspec = snap_dir / "buildspec.json"
    shutil.copy2(request.buildspec_path, snap_buildspec)

    env_artifact = request.buildspec_path.parent / "environment.json"
    if env_artifact.is_file():
        shutil.copy2(env_artifact, snap_dir / "environment.json")

    return _SnapshotArtifacts(
        registry_path=snap_registry,
        schema_manifest_path=snap_manifest,
        buildspec_path=snap_buildspec,
    )


def _build_snapshot_datasets(
    manifest_paths: tuple[Path, ...],
) -> dict[str, SnapshotDatasetEntry]:
    datasets: dict[str, SnapshotDatasetEntry] = {}
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            msg = f"Dataset manifest not found: {manifest_path}"
            raise FileNotFoundError(msg)
        manifest = read_dataset_manifest(manifest_path)
        table_key = manifest.table_key
        if table_key in datasets:
            msg = f"Duplicate dataset manifest entry for {table_key}"
            raise ValueError(msg)
        datasets[table_key] = SnapshotDatasetEntry(
            manifest_path=str(manifest_path.resolve()),
            schema_hash=manifest.schema_hash,
            partition_columns=manifest.partition_columns,
            row_count=manifest.row_count,
            stats=manifest.stats,
        )
    return datasets


def _write_snapshot_manifest(path: Path, manifest: ServingSnapshotManifest) -> None:
    manifest.write_json(path)


def _write_current_pointer(serve_dir: Path, pointer: ServingSnapshotPointer) -> None:
    current_path = serve_dir / "current.json"
    _atomic_write_text(current_path, pointer.to_json() + "\n")


def _prune_snapshots(*, serve_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    snaps_root = serve_dir / "snapshots"
    if not snaps_root.exists():
        return
    dirs = sorted(
        [p for p in snaps_root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in dirs[keep_last:]:
        shutil.rmtree(old, ignore_errors=True)


def publish_serving_snapshot(
    *,
    gateway: SnapshotPublisherGateway,
    request: PublishServingSnapshotRequest,
) -> ServingSnapshotManifest:
    """Publish an immutable serving snapshot.

    Parameters
    ----------
    gateway
        Storage gateway with the build database connection.
    request
        Publish request with snapshot identity, paths, and retention settings.

    Returns
    -------
    ServingSnapshotManifest
        Published snapshot manifest (written to snapshot_manifest.json).

    Raises
    ------
    FileNotFoundError
        If build database not found.
    """
    db_path = gateway.config.db_path
    if not db_path.is_file():
        msg = f"Build DB not found: {db_path}"
        raise FileNotFoundError(msg)

    gateway.execute("CHECKPOINT")
    if gateway.config.attach_meta:
        gateway.execute(f"CHECKPOINT {META_CATALOG_NAME}")
    gateway.con.commit()

    snap_dir = request.serve_dir / "snapshots" / request.run_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    snap_db = _copy_snapshot_database(snap_dir=snap_dir, config=gateway.config)
    artifacts = _copy_snapshot_artifacts(request, snap_dir=snap_dir)
    version = _compute_semantic_version(
        artifacts.registry_path,
        artifacts.schema_manifest_path,
        artifacts.buildspec_path,
    )
    datasets = _build_snapshot_datasets(request.dataset_manifest_paths)
    published_at = datetime.now(UTC)

    manifest = ServingSnapshotManifest(
        run_id=request.run_id,
        repo=gateway.config.repo or "unknown",
        commit=gateway.config.commit or "unknown",
        published_at=published_at.isoformat(),
        db_path=str(snap_db),
        semantic_registry_path=str(artifacts.registry_path),
        schema_manifest_path=str(artifacts.schema_manifest_path),
        buildspec_path=str(artifacts.buildspec_path),
        semantic_layer_version=version,
        datasets=datasets,
    )
    snapshot_manifest_path = snap_dir / "snapshot_manifest.json"
    _write_snapshot_manifest(snapshot_manifest_path, manifest)
    _prepare_snapshot_tables(
        snap_db=snap_db,
        run_id=request.run_id,
        snapshot_manifest=manifest,
    )

    pointer = ServingSnapshotPointer(
        snapshot_root=snap_dir,
        snapshot_manifest_path=snapshot_manifest_path,
        db_path=snap_db,
        semantic_registry_path=artifacts.registry_path,
        schema_manifest_path=artifacts.schema_manifest_path,
        buildspec_path=artifacts.buildspec_path,
        repo=gateway.config.repo or "unknown",
        commit=gateway.config.commit or "unknown",
        run_id=request.run_id,
        published_at=published_at,
        semantic_layer_version=version,
    )

    _write_current_pointer(request.serve_dir, pointer)
    _prune_snapshots(serve_dir=request.serve_dir, keep_last=request.keep_last)

    return manifest


__all__ = ["PublishServingSnapshotRequest", "publish_serving_snapshot"]
