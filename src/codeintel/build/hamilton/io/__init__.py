"""Hamilton IO adapters for dataset materialization.

This package provides:
- DatasetRef: Type-safe dataset references in the Hamilton DAG
- IbisIOConfig: Configuration for Ibis-based IO operations
- @dataloader/@datasaver implementations via IbisGateway
"""

from __future__ import annotations

from codeintel.build.hamilton.io.dataset_ref import DatasetRef, refs_from_target_result
from codeintel.build.hamilton.io.ibis_adapter import IbisIOConfig

__all__ = [
    "DatasetRef",
    "IbisIOConfig",
    "refs_from_target_result",
]
