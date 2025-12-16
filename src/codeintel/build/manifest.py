"""Backward compatibility alias for build manifest data models.

The canonical location for these dataclasses is `codeintel.core.build_manifest`
so that storage can depend on them without importing build.
"""

from __future__ import annotations

from codeintel.core.build_manifest import (
    BuildRunRecord,
    BuildStatus,
    OutputManifest,
)

__all__ = [
    "BuildRunRecord",
    "BuildStatus",
    "OutputManifest",
]
