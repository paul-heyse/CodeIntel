"""Dataset helpers for build-time Arrow dataset IO."""

from __future__ import annotations

from codeintel.core.datasets.arrow_store import (
    ArrowDatasetInput,
    ArrowDatasetManifestRequest,
    ArrowDatasetScanOptions,
    ArrowDatasetStats,
    ArrowDatasetWriteOptions,
    ExistingDataBehavior,
    ScanPushdown,
    build_dataset_manifest,
    dataset_stats,
    scan_dataset,
    scan_dataset_reader,
    scan_dataset_scanner,
    write_dataset,
)
from codeintel.core.datasets.manifests import (
    DATASET_MANIFEST_FILENAME,
    dataset_manifest_path,
    load_dataset_manifest,
    read_dataset_manifest,
    write_dataset_manifest,
)
from codeintel.core.datasets.parquet_metadata import (
    column_types_from_metadata,
    metadata_from_schema,
    table_schema_from_dataset,
    table_schema_from_parquet_metadata,
)
from codeintel.core.datasets.paths import SnapshotIdError, dataset_snapshot_dir, dataset_table_dir

__all__ = [
    "DATASET_MANIFEST_FILENAME",
    "ArrowDatasetInput",
    "ArrowDatasetManifestRequest",
    "ArrowDatasetScanOptions",
    "ArrowDatasetStats",
    "ArrowDatasetWriteOptions",
    "ExistingDataBehavior",
    "ScanPushdown",
    "SnapshotIdError",
    "build_dataset_manifest",
    "column_types_from_metadata",
    "dataset_manifest_path",
    "dataset_snapshot_dir",
    "dataset_stats",
    "dataset_table_dir",
    "load_dataset_manifest",
    "metadata_from_schema",
    "read_dataset_manifest",
    "scan_dataset",
    "scan_dataset_reader",
    "scan_dataset_scanner",
    "table_schema_from_dataset",
    "table_schema_from_parquet_metadata",
    "write_dataset",
    "write_dataset_manifest",
]
