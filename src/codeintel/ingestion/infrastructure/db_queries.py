"""Safe database query helpers for ingestion plugins using Ibis.

This module re-exports query utilities from ``codeintel.storage.queries.safe``,
providing backward compatibility for existing imports.

The canonical implementations now live in ``codeintel.storage.queries``.

Examples
--------
>>> from codeintel.ingestion.infrastructure.db_queries import safe_count
>>> count = safe_count(gateway, "core.ast_nodes")
>>> count
42

See Also
--------
codeintel.storage.queries : Canonical query utility implementations
"""

from __future__ import annotations

# Re-export canonical implementations from storage.queries
from codeintel.storage.queries.safe import (
    DUCKDB_QUERY_ERRORS,
    ColumnNotFoundError,
    ForeignKeyRef,
    QueryError,
    TableNotFoundError,
    safe_count,
    safe_count_duplicates,
    safe_count_non_positive,
    safe_count_nulls,
    safe_count_orphan_refs,
    safe_count_with_scope,
    safe_get_columns,
    safe_max_value,
    safe_min_value,
    safe_not_null_fraction,
    safe_table_exists,
)

__all__ = [
    "DUCKDB_QUERY_ERRORS",
    "ColumnNotFoundError",
    "ForeignKeyRef",
    "QueryError",
    "TableNotFoundError",
    "safe_count",
    "safe_count_duplicates",
    "safe_count_non_positive",
    "safe_count_nulls",
    "safe_count_orphan_refs",
    "safe_count_with_scope",
    "safe_get_columns",
    "safe_max_value",
    "safe_min_value",
    "safe_not_null_fraction",
    "safe_table_exists",
]
