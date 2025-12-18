"""Shared boundary types for the Hamilton build DAG.

These aliases define the canonical shapes that cross Hamilton "execution
boundaries" (e.g., saver metadata dicts, row-count mappings, etc.). Centralizing
them prevents subtle drift between modules that can cause Hamilton's strict type
matching to fail graph compilation.
"""

from __future__ import annotations

from codeintel.storage.helpers.table_key import TableKey

type MaterializationMetadata = dict[str, object]
type RowCounts = dict[str, int]

type TargetName = str

__all__ = [
    "MaterializationMetadata",
    "RowCounts",
    "TableKey",
    "TargetName",
]
