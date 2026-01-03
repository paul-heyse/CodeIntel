"""Compatibility re-exports for Arrow dataset scanning helpers."""

from codeintel.core.datasets.scanning import (
    DatasetScanOptions,
    QueryPlanSpec,
    build_scanner,
    dataset_for_manifest,
    resolve_partitioning,
    unify_dataset_schema,
)

__all__ = [
    "DatasetScanOptions",
    "QueryPlanSpec",
    "build_scanner",
    "dataset_for_manifest",
    "resolve_partitioning",
    "unify_dataset_schema",
]
