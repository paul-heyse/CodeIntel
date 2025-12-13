"""Artifact materializer utilities for native Hamilton targets.

This module provides utilities for materializing file-based artifacts (exports,
reports, indexes) with atomic write semantics and proper ArtifactRef generation.

The ArtifactMaterializationContext is compatible with BuildContext from the
unified context hierarchy, enabling seamless integration with the consolidated
build system.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer
from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.storage.tracking.asset_tracking import AssetRecord

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from codeintel.build.context_base import BuildContext
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway.protocol import StorageGateway


@dataclass(frozen=True)
class ArtifactMaterializationContext:
    """Context for artifact materialization.

    This context is compatible with BuildContext from the unified hierarchy.
    Both can be used interchangeably for artifact materialization operations.

    Attributes
    ----------
    snapshot
        Snapshot reference (repo, commit) for lineage tracking.
    gateway
        Optional storage gateway for asset catalog recording.
    owner_target
        Optional target name that produced these assets (for asset catalog).
    input_hash
        Optional input hash from manifest (for asset catalog).
    """

    snapshot: SnapshotRef
    gateway: StorageGateway | None = None
    owner_target: str | None = None
    input_hash: str | None = None

    @classmethod
    def from_build_context(
        cls,
        ctx: BuildContext,
        *,
        owner_target: str | None = None,
        input_hash: str | None = None,
    ) -> ArtifactMaterializationContext:
        """Create ArtifactMaterializationContext from a BuildContext.

        Parameters
        ----------
        ctx
            BuildContext with gateway and snapshot.
        owner_target
            Target name that produced these assets.
        input_hash
            Input hash for asset catalog.

        Returns
        -------
        ArtifactMaterializationContext
            New artifact materialization context.
        """
        return cls(
            snapshot=ctx.snapshot,
            gateway=ctx.gateway,
            owner_target=owner_target,
            input_hash=input_hash,
        )


@dataclass(frozen=True)
class ArtifactMaterializationSpec:
    """Artifact specification for materialization."""

    artifact_name: str
    artifact_type: str
    content: bytes | str
    output_path: Path
    metadata: dict[str, object] | None = None


def materialize_artifact(
    ctx: ArtifactMaterializationContext,
    spec: ArtifactMaterializationSpec,
) -> ArtifactRef:
    """Write artifact to disk with atomic semantics and return ArtifactRef.

    Uses a temp file + atomic rename pattern to ensure the artifact is either
    fully written or not present at all. This prevents partial writes from
    corrupting the build artifacts.

    Parameters
    ----------
    ctx
        Materialization context.
    spec
        Artifact materialization specification.

    Returns
    -------
    ArtifactRef
        Reference to the materialized artifact with validated path.

    Examples
    --------
    >>> from pathlib import Path
    >>> from codeintel.config.primitives import SnapshotRef
    >>> snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=Path("/tmp"))
    >>> ctx = ArtifactMaterializationContext(snapshot=snapshot)
    >>> spec = ArtifactMaterializationSpec(
    ...     artifact_name="test_export",
    ...     artifact_type="file",
    ...     content=b"test data",
    ...     output_path=Path("/tmp/export.json"),
    ... )
    >>> ref = materialize_artifact(ctx, spec)
    >>> ref.name
    'test_export'
    """
    # Validate contract if strict mode is enabled
    ContractEnforcer.validate_artifact_write(spec.artifact_name)

    LOG.info(
        "Materializing artifact '%s' to %s (%d bytes)",
        spec.artifact_name,
        spec.output_path,
        len(spec.content) if isinstance(spec.content, bytes) else len(spec.content.encode("utf-8")),
    )

    # Ensure parent directory exists
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert string to bytes if needed
    content_bytes = spec.content.encode("utf-8") if isinstance(spec.content, str) else spec.content

    # Write to temporary file in the same directory (for atomic rename)
    temp_fd, temp_path_str = tempfile.mkstemp(
        dir=spec.output_path.parent,
        prefix=f".{spec.output_path.name}.",
        suffix=".tmp",
    )

    temp_path = Path(temp_path_str)

    try:
        # Write content to temp file
        with os.fdopen(temp_fd, "wb") as f:
            f.write(content_bytes)

        # Atomic rename to final location
        temp_path.rename(spec.output_path)

        LOG.info(
            "Successfully materialized artifact '%s' to %s",
            spec.artifact_name,
            spec.output_path,
        )

    except Exception:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise

    # Create ArtifactRef with metadata
    artifact_metadata = {
        "description": f"Materialized {spec.artifact_type} artifact",
        "size_bytes": len(content_bytes),
        **(spec.metadata or {}),
    }

    artifact_ref = ArtifactRef(
        name=spec.artifact_name,
        artifact_type=spec.artifact_type,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        path=str(spec.output_path),
        metadata=artifact_metadata,
    )

    # Record in asset catalog if gateway and owner_target provided
    if ctx.gateway is not None and ctx.owner_target is not None:
        ctx.gateway.assets.record_asset(
            AssetRecord(
                asset_key=spec.artifact_name,
                asset_type="artifact",
                repo=ctx.snapshot.repo,
                commit=ctx.snapshot.commit,
                owner_target=ctx.owner_target,
                file_size_bytes=len(content_bytes),
                input_hash=ctx.input_hash,
                metadata=artifact_metadata,
            )
        )

    return artifact_ref


def materialize_artifacts(
    artifacts: dict[str, tuple[str, bytes | str, Path]],
    ctx: ArtifactMaterializationContext,
    metadata_by_name: dict[str, dict[str, object]] | None = None,
) -> tuple[ArtifactRef, ...]:
    r"""Materialize multiple artifacts atomically.

    Parameters
    ----------
    artifacts
        Dictionary mapping artifact names to (artifact_type, content, output_path) tuples.
    ctx
        Materialization context.
    metadata_by_name
        Optional mapping of artifact name to metadata.

    Returns
    -------
    tuple[ArtifactRef, ...]
        Tuple of ArtifactRef objects for all materialized artifacts.

    Examples
    --------
    >>> artifacts = {
    ...     "export_jsonl": ("file", b'{"data": "value"}', Path("/tmp/export.jsonl")),
    ...     "export_csv": ("file", b"col1,col2\\n1,2", Path("/tmp/export.csv")),
    ... }
    >>> refs = materialize_artifacts(artifacts, ArtifactMaterializationContext(snapshot=snapshot))
    >>> len(refs)
    2
    """
    artifact_refs: list[ArtifactRef] = []

    for name, (artifact_type, content, output_path) in artifacts.items():
        spec = ArtifactMaterializationSpec(
            artifact_name=name,
            artifact_type=artifact_type,
            content=content,
            output_path=output_path,
            metadata=metadata_by_name.get(name) if metadata_by_name else None,
        )
        ref = materialize_artifact(ctx, spec)
        artifact_refs.append(ref)

    return tuple(artifact_refs)


__all__ = [
    "ArtifactMaterializationContext",
    "ArtifactMaterializationSpec",
    "materialize_artifact",
    "materialize_artifacts",
]
