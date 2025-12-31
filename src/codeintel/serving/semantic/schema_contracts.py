"""Dataset-backed schema resolution for serving engines."""

from __future__ import annotations

import pyarrow as pa

from codeintel.serving.semantic.datasets import (
    DatasetManifestEntry,
    DatasetManifestIndex,
    dataset_for_entry,
)
from codeintel.storage.datasets.contracts import arrow_schema_from_manifest


def contract_schema_for_table_key(
    *,
    dataset_manifests: DatasetManifestIndex,
    table_key: str,
) -> pa.Schema | None:
    """Return a contract schema for a table key from dataset manifests.

    Returns
    -------
    pa.Schema | None
        The resolved contract schema, or None when unavailable.
    """
    entry = dataset_manifests.get(table_key)
    if entry is None:
        return None
    try:
        return _contract_schema_from_entry(entry)
    except (TypeError, ValueError):
        return None


def _contract_schema_from_entry(entry: DatasetManifestEntry) -> pa.Schema | None:
    schema = arrow_schema_from_manifest(entry.manifest)
    if schema is not None:
        return schema
    try:
        dataset = dataset_for_entry(entry)
    except (OSError, ValueError, pa.ArrowInvalid):
        return None
    return dataset.schema


__all__ = ["contract_schema_for_table_key"]
