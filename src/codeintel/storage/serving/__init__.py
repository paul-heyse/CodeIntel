"""Serving-related storage utilities.

This package contains storage-owned helpers for building and querying serving
artifacts (e.g., search indices) without leaking DuckDB implementation details
into higher application layers.
"""

from __future__ import annotations

from codeintel.storage.serving.search_index import build_search_documents_table, ensure_fts_index
from codeintel.storage.serving.snapshot_service import (
    LineageMetadataError,
    SearchIndexBuildError,
    ServingSnapshotError,
    ServingSnapshotService,
)

__all__ = [
    "LineageMetadataError",
    "SearchIndexBuildError",
    "ServingSnapshotError",
    "ServingSnapshotService",
    "build_search_documents_table",
    "ensure_fts_index",
]
