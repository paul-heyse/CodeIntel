"""Typed snapshot identity models for serving.

These models provide a single source of truth for snapshot identity across
HTTP and FastMCP adapters while keeping serialized JSON stable.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from codeintel.serving.operations.protocols import ServingSnapshotPointerProtocol


class ServingSnapshotIdentity(BaseModel):
    """Minimal snapshot identity used across query/search responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str
    commit: str
    run_id: str

    @classmethod
    def from_pointer(cls, pointer: ServingSnapshotPointerProtocol) -> ServingSnapshotIdentity:
        """Create a minimal identity model from a snapshot pointer.

        Returns
        -------
        ServingSnapshotIdentity
            Snapshot identity derived from the pointer.
        """
        return cls(repo=pointer.repo, commit=pointer.commit, run_id=pointer.run_id)


class ServingSnapshotRef(ServingSnapshotIdentity):
    """Expanded snapshot identity used for discovery/meta surfaces."""

    published_at: datetime

    @classmethod
    def from_pointer(cls, pointer: ServingSnapshotPointerProtocol) -> ServingSnapshotRef:
        """Create a snapshot reference model from a snapshot pointer.

        Returns
        -------
        ServingSnapshotRef
            Snapshot reference derived from the pointer.
        """
        return cls(
            repo=pointer.repo,
            commit=pointer.commit,
            run_id=pointer.run_id,
            published_at=pointer.published_at,
        )


class ServingExportSnapshot(BaseModel):
    """Snapshot identity stored alongside export artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str
    commit: str
    run_id: str
    published_at: datetime
    semantic_layer_hash: str

    @classmethod
    def from_pointer(cls, pointer: ServingSnapshotPointerProtocol) -> ServingExportSnapshot:
        """Create an export snapshot model from a snapshot pointer.

        Returns
        -------
        ServingExportSnapshot
            Export snapshot metadata derived from the pointer.
        """
        return cls(
            repo=pointer.repo,
            commit=pointer.commit,
            run_id=pointer.run_id,
            published_at=pointer.published_at,
            semantic_layer_hash=pointer.semantic_layer_version,
        )


__all__ = ["ServingExportSnapshot", "ServingSnapshotIdentity", "ServingSnapshotRef"]
