"""Data existence checking utilities for the storage layer.

This module provides functions to check whether dataset tables contain
data for specific repository/commit snapshots. These utilities are used
by the auto-pipeline to determine if prerequisite data exists.

The functions in this module are designed to be efficient, using LIMIT 1
queries to minimize database overhead.
"""

from __future__ import annotations

from codeintel.storage.queries.safe import (
    count_rows_for_snapshot,
    count_rows_for_tables,
    safe_count_rows,
    table_has_rows_for_snapshot,
)

__all__ = [
    "count_rows_for_snapshot",
    "count_rows_for_tables",
    "safe_count_rows",
    "table_has_rows_for_snapshot",
]
