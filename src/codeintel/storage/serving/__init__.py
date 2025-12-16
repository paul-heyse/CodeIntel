"""Serving-related storage utilities.

This package contains storage-owned helpers for building and querying serving
artifacts (e.g., search indices) without leaking DuckDB implementation details
into higher application layers.
"""

from __future__ import annotations

from codeintel.storage.serving.search_index import build_search_documents_table, ensure_fts_index

__all__ = [
    "build_search_documents_table",
    "ensure_fts_index",
]
