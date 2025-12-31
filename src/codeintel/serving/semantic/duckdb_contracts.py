"""DuckDB-backed contract resolution helpers for serving engines.

This module is a thin wrapper around the storage-level contract resolver to
avoid storage-layer dependencies on serving code.
"""

from __future__ import annotations

from codeintel.storage.schema.duckdb_contracts import (
    contract_schema_for_table_key,
    table_schema_for_table_key,
)

__all__ = ["contract_schema_for_table_key", "table_schema_for_table_key"]
