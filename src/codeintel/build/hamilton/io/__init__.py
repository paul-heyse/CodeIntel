"""Hamilton IO adapters for dataset materialization.

This package provides:
- DatasetRef: Type-safe dataset references in the Hamilton DAG
- ArtifactRef: Type-safe artifact references for non-tabular outputs
- Relation-first IO helpers for DuckDB-backed datasets
"""

from __future__ import annotations

from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.io.dataset_ref import DatasetRef, refs_from_target_result
from codeintel.build.hamilton.io.duckdb_relation_adapter import load_dataset_relation

__all__ = [
    "ArtifactRef",
    "DatasetRef",
    "load_dataset_relation",
    "refs_from_target_result",
]
