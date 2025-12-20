"""Metadata catalog and bootstrap utilities."""

from __future__ import annotations

from codeintel.storage.metadata.ddl import apply_metadata_ddl
from codeintel.storage.metadata.sync import (
    bootstrap_metadata_datasets,
    load_dataset_schema_registry,
    load_derived_lineage_columns,
    sync_dataset_dataflow_graph,
    sync_derived_lineage_columns,
    validate_dataset_schema_registry,
)

__all__ = [
    "apply_metadata_ddl",
    "bootstrap_metadata_datasets",
    "load_dataset_schema_registry",
    "load_derived_lineage_columns",
    "sync_dataset_dataflow_graph",
    "sync_derived_lineage_columns",
    "validate_dataset_schema_registry",
]
