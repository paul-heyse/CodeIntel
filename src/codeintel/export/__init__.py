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

from codeintel.build.schemas import iter_contracts


def default_validation_schemas() -> list[str]:
    """Return the set of dataset names that should be validated by default.

    Derived from contracts in the build.schemas contract provider.

    Returns
    -------
    list[str]
        Sorted dataset names with JSON Schema validation configured.
    """
    return sorted(c.name for c in iter_contracts() if c.json_schema_id is not None)


__all__ = ["default_validation_schemas"]
