"""DuckDB full-text search (FTS) index helpers for serving snapshots.

DuckDB-specific implementation details are owned by ``codeintel.storage``. This
module remains as a thin import surface for build/serving code.
"""

from __future__ import annotations

from codeintel.storage.serving.search_index import build_search_documents_table, ensure_fts_index

__all__ = ["build_search_documents_table", "ensure_fts_index"]
