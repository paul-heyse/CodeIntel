"""Serving snapshot publisher.

Publishes immutable read-only snapshots from the build database with atomic
pointer updates for zero-downtime deployments.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Protocol

from codeintel.build.serving.manifest import ServingSnapshotManifest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.gateway.protocol import DuckDBConnection


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
    keep_last
        Number of old snapshots to retain.
    """

    run_id: str
    serve_dir: Path
    semantic_registry_path: Path
    schema_manifest_path: Path
    keep_last: int = 10


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
) -> str:
    """Compute semantic layer version hash.

    Returns
    -------
    str
        Short stable hash derived from registry and manifest bytes.
    """
    hasher = hashlib.sha256()
    for p in (registry_path, manifest_path):
        if p.exists():
            hasher.update(p.read_bytes())
    return hasher.hexdigest()[:16]


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
        Published snapshot manifest (also written to current.json).

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
    gateway.con.commit()

    snap_dir = request.serve_dir / "snapshots" / request.run_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    snap_db = snap_dir / "codeintel.duckdb"
    shutil.copy2(db_path, snap_db)

    snap_registry = snap_dir / "semantic_registry.json"
    shutil.copy2(request.semantic_registry_path, snap_registry)

    snap_manifest = snap_dir / "schema_manifest.json"
    shutil.copy2(request.schema_manifest_path, snap_manifest)

    version = _compute_semantic_version(snap_registry, snap_manifest)

    manifest = ServingSnapshotManifest(
        run_id=request.run_id,
        repo=gateway.config.repo or "unknown",
        commit=gateway.config.commit or "unknown",
        published_at=datetime.now(UTC).isoformat(),
        db_path=str(snap_db),
        semantic_registry_path=str(snap_registry),
        schema_manifest_path=str(snap_manifest),
        semantic_layer_version=version,
    )

    current_path = request.serve_dir / "current.json"
    _atomic_write_text(current_path, manifest.to_json() + "\n")

    if request.keep_last > 0:
        snaps_root = request.serve_dir / "snapshots"
        if snaps_root.exists():
            dirs = sorted(
                [p for p in snaps_root.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old in dirs[request.keep_last :]:
                shutil.rmtree(old, ignore_errors=True)

    return manifest


__all__ = ["PublishServingSnapshotRequest", "publish_serving_snapshot"]
