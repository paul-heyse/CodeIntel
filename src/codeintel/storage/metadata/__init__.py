"""Metadata catalog and bootstrap utilities.

This package provides utilities for managing the CodeIntel metadata catalog:

- metadata.bootstrap: DDL definitions and bootstrap logic for metadata tables
"""

from __future__ import annotations

from codeintel.storage.metadata.bootstrap import (
    AUTO_NORMALIZED_MACRO_DDLS,
    INGEST_MACRO_DDLS,
    INGEST_MACROS,
    METADATA_SCHEMA_DDL,
    METADATA_SCHEMA_DDL_BASE,
    METADATA_SCHEMA_DDL_REST,
    NORMALIZED_MACROS,
    PIPELINE_INDEXES_DDL,
    PIPELINE_RUNS_DDL,
    PIPELINE_STEPS_DDL,
    _assert_macro_coverage,
    apply_metadata_ddl,
    bootstrap_metadata_datasets,
    dataset_rows_only_entries,
    ingest_macro_coverage,
    load_dataset_schema_registry,
    load_macro_registry,
    sync_dataset_dataflow_graph,
    validate_dataset_schema_registry,
    validate_macro_registry,
    validate_normalized_macro_schemas,
)

__all__ = [
    "AUTO_NORMALIZED_MACRO_DDLS",
    "INGEST_MACROS",
    "INGEST_MACRO_DDLS",
    "METADATA_SCHEMA_DDL",
    "METADATA_SCHEMA_DDL_BASE",
    "METADATA_SCHEMA_DDL_REST",
    "NORMALIZED_MACROS",
    "PIPELINE_INDEXES_DDL",
    "PIPELINE_RUNS_DDL",
    "PIPELINE_STEPS_DDL",
    "_assert_macro_coverage",
    "apply_metadata_ddl",
    "bootstrap_metadata_datasets",
    "dataset_rows_only_entries",
    "ingest_macro_coverage",
    "load_dataset_schema_registry",
    "load_macro_registry",
    "sync_dataset_dataflow_graph",
    "validate_dataset_schema_registry",
    "validate_macro_registry",
    "validate_normalized_macro_schemas",
]
