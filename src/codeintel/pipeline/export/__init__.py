"""Doc export utilities for emitting CodeIntel analytics as JSONL or Parquet artifacts."""

from __future__ import annotations

from codeintel.storage.datasets import JSON_SCHEMA_BY_DATASET_NAME


def default_validation_schemas() -> list[str]:
    """
    Return the set of dataset names that should be validated by default.

    Derived from JSON_SCHEMA_BY_DATASET_NAME in the dataset contract layer.

    Returns
    -------
    list[str]
        Sorted dataset names with JSON Schema validation configured.
    """
    return sorted(JSON_SCHEMA_BY_DATASET_NAME.keys())


# Backwards-compatible constant; prefer calling default_validation_schemas().
DEFAULT_VALIDATION_SCHEMAS: list[str] = default_validation_schemas()
