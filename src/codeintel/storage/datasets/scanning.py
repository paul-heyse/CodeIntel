"""Compatibility re-exports for Arrow dataset scanning helpers."""

from codeintel.core.columnar.streaming import (
    DatasetScanOptions,
    QueryPlanSpec,
    dataset_for_manifest,
    resolve_partitioning,
    unify_dataset_schema,
)
from codeintel.core.datasets.scanner_ops import build_scanner
from codeintel.core.datasets.scanning import (
    ParquetScanOptions,
    ParquetScanTelemetry,
    scan_parquet_dataset,
    scan_parquet_dataset_with_telemetry,
    scan_parquet_table,
)

__all__ = [
    "DatasetScanOptions",
    "ParquetScanOptions",
    "ParquetScanTelemetry",
    "QueryPlanSpec",
    "build_scanner",
    "dataset_for_manifest",
    "resolve_partitioning",
    "scan_parquet_dataset",
    "scan_parquet_dataset_with_telemetry",
    "scan_parquet_table",
    "unify_dataset_schema",
]
