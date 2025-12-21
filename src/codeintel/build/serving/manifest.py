"""Serving snapshot manifest dataclass.

The manifest is written to `current.json` as an atomic pointer to the active
serving snapshot.
"""

from codeintel.core.manifests import ServingSnapshotManifest

__all__ = ["ServingSnapshotManifest"]
