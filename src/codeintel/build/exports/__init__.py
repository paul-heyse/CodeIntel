"""Build system export infrastructure for JSONL and Parquet artifacts.

This package provides unified export functionality. Exports are managed as
build targets integrated with the Hamilton DAG.

Usage
-----
>>> from codeintel.build.exports import ExportCallOptions, export_all_jsonl
>>> options = ExportCallOptions(validate_exports=True, datasets=["function_metrics"])
>>> export_all_jsonl(gateway, output_dir, options=options)
"""

from __future__ import annotations

from codeintel.build.exports.common import (
    AuditRecord,
    ExportCallOptions,
    ExportTarget,
    default_validation_schemas,
)
from codeintel.build.exports.jsonl import (
    export_all_jsonl,
    export_dataset_to_jsonl,
    export_jsonl_for_table,
    export_repo_map_json,
)
from codeintel.build.exports.manifest import (
    ExportManifestData,
    IncrementalMarker,
    SkipCriteria,
    compute_file_hash,
    read_incremental_marker,
    should_skip_export,
    write_dataset_manifest,
    write_incremental_marker,
    write_per_dataset_manifest,
)
from codeintel.build.exports.parquet import (
    export_all_parquet,
    export_dataset_to_parquet,
    export_parquet_for_table,
)
from codeintel.build.exports.runner import (
    Exporter,
    ExportOptions,
    ExportRunner,
    JsonlExporter,
    run_validated_exports,
)
from codeintel.build.exports.validation import validate_export_files

__all__ = [
    "AuditRecord",
    "ExportCallOptions",
    "ExportManifestData",
    "ExportOptions",
    "ExportRunner",
    "ExportTarget",
    "Exporter",
    "IncrementalMarker",
    "JsonlExporter",
    "SkipCriteria",
    "compute_file_hash",
    "default_validation_schemas",
    "export_all_jsonl",
    "export_all_parquet",
    "export_dataset_to_jsonl",
    "export_dataset_to_parquet",
    "export_jsonl_for_table",
    "export_parquet_for_table",
    "export_repo_map_json",
    "read_incremental_marker",
    "run_validated_exports",
    "should_skip_export",
    "validate_export_files",
    "write_dataset_manifest",
    "write_incremental_marker",
    "write_per_dataset_manifest",
]
