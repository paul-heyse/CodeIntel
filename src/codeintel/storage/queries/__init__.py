"""Safe database query utilities.

This package provides type-safe query helpers for database operations,
using DuckDB relations and handling errors gracefully.

Key Components
--------------
safe_count
    Safely count rows in a table.
safe_table_exists
    Check if a table exists.
safe_count_nulls
    Count NULL values in a column.
ForeignKeyRef
    Foreign key reference specification.

Example
-------
```python
from codeintel.storage.queries import safe_count, safe_table_exists

if safe_table_exists(gateway, "core.modules"):
    count = safe_count(gateway, "core.modules")
```
"""

from __future__ import annotations

from codeintel.core.queries.safe import (
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
