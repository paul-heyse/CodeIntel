"""SQL query building utilities for DuckDB operations.

This package provides type-safe SQL construction utilities:

- sql.primitives: Core SQL building with no external dependencies
- sql.builder: Schema-aware functions that integrate with codeintel.config.datasets
  (import directly from sql.builder to avoid circular imports)

For most use cases, import directly from this package which re-exports all
public symbols from primitives. For schema-aware functions like
`ensure_schema` and `prepared_statements_dynamic`, import directly from
`codeintel.storage.sql.builder`.
"""

from __future__ import annotations

# Only re-export from primitives to avoid circular imports with config.datasets
# Schema-aware functions in builder.py should be imported directly from
# codeintel.storage.sql.builder
from codeintel.storage.sql.primitives import (
    TABLE_KEY_PARTS,
    InvalidIdentifierError,
    PreparedStatements,
    QueryBuilder,
    SafeColumn,
    SafeTable,
    SqlBuilderError,
    SqlParams,
    build_delete_query,
    build_insert_sql,
    macro_select_sql,
    quote_identifier,
    quote_table_key,
    render_sql,
    safe_macro_call,
    validate_identifier,
)

__all__ = [
    "TABLE_KEY_PARTS",
    "InvalidIdentifierError",
    "PreparedStatements",
    "QueryBuilder",
    "SafeColumn",
    "SafeTable",
    "SqlBuilderError",
    "SqlParams",
    "build_delete_query",
    "build_insert_sql",
    "macro_select_sql",
    "quote_identifier",
    "quote_table_key",
    "render_sql",
    "safe_macro_call",
    "validate_identifier",
]
