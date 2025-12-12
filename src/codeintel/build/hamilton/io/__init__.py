"""Hamilton IO adapters for dataset materialization.

This package provides:
- DatasetRef: Type-safe dataset references in the Hamilton DAG
- ArtifactRef: Type-safe artifact references for non-tabular outputs
- IbisIOConfig: Configuration for Ibis-based IO operations
- @dataloader/@datasaver implementations via IbisGateway
"""

from __future__ import annotations

from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.io.dataset_ref import DatasetRef, refs_from_target_result
from codeintel.build.hamilton.io.ibis_adapter import (
    IbisIOConfig,
    load_dataset_df,
    load_dataset_ibis,
)

__all__ = [
    "ArtifactRef",
    "DatasetRef",
    "IbisIOConfig",
    "load_dataset_df",
    "load_dataset_ibis",
    "refs_from_target_result",
]
