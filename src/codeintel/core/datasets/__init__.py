"""Dataset helpers for build-time Arrow dataset IO.

This package avoids eager imports to prevent circular import chains when
downstream modules only need lightweight metadata helpers.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Final

_MODULE_EXPORTS: Final[dict[str, tuple[str, ...]]] = {
    "codeintel.core.datasets.arrow_store": (
        "ArrowDatasetInput",
        "ArrowDatasetManifestRequest",
        "ArrowDatasetScanOptions",
        "ArrowDatasetStats",
        "ArrowDatasetWriteOptions",
        "ExistingDataBehavior",
        "ScanPushdown",
        "build_dataset_manifest",
        "dataset_stats",
        "scan_dataset",
        "scan_dataset_reader",
        "scan_dataset_scanner",
        "write_dataset",
    ),
    "codeintel.core.datasets.manifests": (
        "DATASET_MANIFEST_FILENAME",
        "dataset_manifest_path",
        "load_dataset_manifest",
        "read_dataset_manifest",
        "write_dataset_manifest",
    ),
    "codeintel.core.datasets.parquet_metadata": (
        "column_types_from_metadata",
        "metadata_from_schema",
        "table_schema_from_dataset",
        "table_schema_from_parquet_metadata",
    ),
    "codeintel.core.datasets.paths": (
        "SnapshotIdError",
        "dataset_snapshot_dir",
        "dataset_table_dir",
    ),
}

_EXPORT_TO_MODULE: Final[dict[str, str]] = {
    name: module for module, names in _MODULE_EXPORTS.items() for name in names
}

__all__: Final[tuple[str, ...]] = (
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
)

if TYPE_CHECKING:
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
    from codeintel.core.datasets.paths import (
        SnapshotIdError,
        dataset_snapshot_dir,
        dataset_table_dir,
    )


def __getattr__(name: str) -> object:
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module = importlib.import_module(module_name)
    return getattr(module, name)
