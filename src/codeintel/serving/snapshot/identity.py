"""Snapshot identity conversion helpers.

Prefer typed models from ``codeintel.serving.snapshot.models`` for new code.
This module remains as a thin compatibility layer for callers that need
simple JSON-friendly dictionaries.
"""

from __future__ import annotations

from codeintel.serving.operations.protocols import ServingSnapshotPointerProtocol
from codeintel.serving.snapshot.models import (
    ServingExportSnapshot,
    ServingSnapshotIdentity,
    ServingSnapshotRef,
)


def snapshot_identity_dict(pointer: ServingSnapshotPointerProtocol) -> dict[str, str]:
    """Return the minimal stable snapshot identity dict used in query/search responses."""
    return ServingSnapshotIdentity.from_pointer(pointer).model_dump(mode="json")


def snapshot_ref_dict(pointer: ServingSnapshotPointerProtocol) -> dict[str, str]:
    """Return the expanded snapshot dict used by MCP `SnapshotRef` models."""
    return ServingSnapshotRef.from_pointer(pointer).model_dump(mode="json")


def export_snapshot_dict(pointer: ServingSnapshotPointerProtocol) -> dict[str, str]:
    """Return the export snapshot dict stored in export metadata sidecars."""
    return ServingExportSnapshot.from_pointer(pointer).model_dump(mode="json")


__all__ = ["export_snapshot_dict", "snapshot_identity_dict", "snapshot_ref_dict"]
