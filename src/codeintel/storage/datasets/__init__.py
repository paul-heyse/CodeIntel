"""Dataset registry and catalog utilities.

This package provides utilities for managing the CodeIntel dataset registry:

- datasets.registry: DatasetRegistry and loading utilities (no circular deps)

Note: Only registry utilities are re-exported here to avoid circular imports.
"""

from __future__ import annotations

from codeintel.storage.datasets.registry import (
    DatasetRegistry,
    build_dataset_dependency_graph,
    dataset_for_name,
    dataset_for_table,
    describe_all_datasets,
    describe_dataset,
    list_dataset_specs,
    load_dataset_registry,
)

__all__ = [
    "DatasetRegistry",
    "build_dataset_dependency_graph",
    "dataset_for_name",
    "dataset_for_table",
    "describe_all_datasets",
    "describe_dataset",
    "list_dataset_specs",
    "load_dataset_registry",
]
