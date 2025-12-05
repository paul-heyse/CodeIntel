"""Export utilities for emitting CodeIntel analytics as JSONL or Parquet artifacts.

This package provides:

- **JSONL export**: Export datasets to JSON Lines format via :func:`export_all_jsonl`
- **Parquet export**: Export datasets to Parquet format via :func:`export_all_parquet`
- **Validation**: Validate exported files against JSON Schema definitions
- **Manifest generation**: Track export metadata and incremental markers

Build System Integration
------------------------
Export operations are available as build targets:

- ``export_jsonl``: Export all datasets to JSONL
- ``export_parquet``: Export all datasets to Parquet

Use ``codeintel build run export_jsonl`` or ``codeintel docs export`` for execution.

ExportCallOptions
-----------------
Import :class:`ExportCallOptions` from :mod:`codeintel.export.export_jsonl` for
programmatic control over export options (validation, datasets, etc.)::

    from codeintel.export.export_jsonl import ExportCallOptions

    options = ExportCallOptions(
        validate_exports=True,
        datasets=["functions", "modules"],
    )
"""

from __future__ import annotations

from codeintel.config.datasets import JSON_SCHEMA_BY_DATASET_NAME


def default_validation_schemas() -> list[str]:
    """Return the set of dataset names that should be validated by default.

    Derived from JSON_SCHEMA_BY_DATASET_NAME in the dataset contract layer.

    Returns
    -------
    list[str]
        Sorted dataset names with JSON Schema validation configured.
    """
    return sorted(JSON_SCHEMA_BY_DATASET_NAME.keys())


__all__ = ["default_validation_schemas"]
