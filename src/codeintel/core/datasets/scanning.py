"""Shared Arrow dataset scanning helpers.

Deprecated: use ``codeintel.core.columnar.streaming``.
"""

from __future__ import annotations

from codeintel.core.columnar.streaming import (
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
