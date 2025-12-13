"""Compatibility re-exports for build artifact references.

Historically, some build submodules imported :class:`ArtifactRef` from
``codeintel.build.artifacts``. The canonical type now lives under the
Hamilton IO layer.
"""

from __future__ import annotations

from codeintel.build.hamilton.io.artifact_ref import ArtifactRef

__all__ = ["ArtifactRef"]
