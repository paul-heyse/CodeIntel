"""Hamilton IO adapters for dataset materialization.

This package provides:
- DatasetRef: Type-safe dataset references in the Hamilton DAG
- ArtifactRef: Type-safe artifact references for non-tabular outputs
"""

from __future__ import annotations

from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.io.dataset_ref import DatasetRef, refs_from_target_result

__all__ = [
    "ArtifactRef",
    "DatasetRef",
    "refs_from_target_result",
]
