"""Snapshot identity helpers for serving.

This package centralizes conversion of serving snapshot pointers into stable,
JSON-friendly identity dictionaries used across serving surfaces.
"""

from codeintel.serving.snapshot.identity import (
    export_snapshot_dict,
    snapshot_identity_dict,
    snapshot_ref_dict,
)
from codeintel.serving.snapshot.models import (
    ServingExportSnapshot,
    ServingSnapshotIdentity,
    ServingSnapshotRef,
)

__all__ = [
    "ServingExportSnapshot",
    "ServingSnapshotIdentity",
    "ServingSnapshotRef",
    "export_snapshot_dict",
    "snapshot_identity_dict",
    "snapshot_ref_dict",
]
