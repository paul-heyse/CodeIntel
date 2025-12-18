"""Metadata catalog and bootstrap utilities.

This package provides utilities for managing the CodeIntel metadata catalog:

- metadata.bootstrap: DDL definitions and bootstrap logic for metadata tables
"""

from __future__ import annotations

from codeintel.storage.metadata.bootstrap import (
    apply_metadata_ddl,
    bootstrap_metadata_datasets,
    load_dataset_schema_registry,
    sync_dataset_dataflow_graph,
    validate_dataset_schema_registry,
)

__all__ = [
    "apply_metadata_ddl",
    "bootstrap_metadata_datasets",
    "load_dataset_schema_registry",
    "sync_dataset_dataflow_graph",
    "validate_dataset_schema_registry",
]
