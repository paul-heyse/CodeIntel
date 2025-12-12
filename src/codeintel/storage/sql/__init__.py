"""SQL query building utilities for DuckDB operations.

This package provides type-safe SQL construction utilities:

- sql.primitives: Core SQL building with no external dependencies
- sql.builder: Schema-aware functions that integrate with codeintel.config.datasets
  (import directly from sql.builder to avoid circular imports)

.. deprecated::
    Most utilities in this module are deprecated. Use ``DuckDBPolicyBackend``
    from ``codeintel.storage.duckdb_policy_backend`` for all SQL operations.

    **Deprecated (use DuckDBPolicyBackend instead):**

    - ``SafeTable``, ``SafeColumn``: Use policy backend methods
    - ``QueryBuilder``: Use ``DuckDBPolicyBackend.bulk_insert()``,
      ``DuckDBPolicyBackend.delete_for_snapshot()``, etc.
    - ``build_insert_sql``, ``build_delete_query``: Use policy backend methods

    **Still supported:**

    - ``quote_identifier``, ``quote_table_key``: Used by the policy backend and storage utilities
    - ``InvalidIdentifierError``, ``SqlBuilderError``: Exception types

For most use cases, import directly from this package which re-exports all
public symbols from primitives. For schema-aware functions like
`ensure_schema` and `prepared_statements_dynamic`, import directly from
`codeintel.storage.sql.builder`.
"""

from __future__ import annotations

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
    quote_identifier,
    quote_table_key,
    render_sql,
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
    "quote_identifier",
    "quote_table_key",
    "render_sql",
    "validate_identifier",
]
