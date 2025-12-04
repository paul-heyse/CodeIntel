"""Storage helper utilities.

This package provides various helper functions:

- helpers.db: Row counts and bulk insertion helpers (no gateway dependencies)
- helpers.profiling: Docs view profiling utilities (imports from gateway)
- helpers.module_index: Module metadata helpers (imports from ingestion)

Note: Only db helpers are re-exported here to avoid circular imports.
Import profiling and module_index directly from their submodules.
"""

from __future__ import annotations

from codeintel.storage.helpers.db import (
    DUCKDB_ERRORS,
    macro_insert_rows,
    row_counts_for_tables,
    safe_row_counts,
)

__all__ = [
    "DUCKDB_ERRORS",
    "macro_insert_rows",
    "row_counts_for_tables",
    "safe_row_counts",
]
