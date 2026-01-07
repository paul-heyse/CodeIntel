"""Compatibility re-exports for Arrow dataset scanning helpers."""

from codeintel.core.columnar.streaming import (
    DatasetScanOptions,
    QueryPlanSpec,
    dataset_for_manifest,
    resolve_partitioning,
    unify_dataset_schema,
)
from codeintel.core.datasets.scanner_ops import build_scanner

__all__ = [
    "DatasetScanOptions",
    "QueryPlanSpec",
    "build_scanner",
    "dataset_for_manifest",
    "resolve_partitioning",
    "unify_dataset_schema",
]
