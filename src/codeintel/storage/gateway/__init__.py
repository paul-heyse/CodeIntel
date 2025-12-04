"""Gateway package for DuckDB storage access.

This package provides the StorageGateway protocol and related types for
accessing CodeIntel DuckDB databases.
"""

from __future__ import annotations

from codeintel.storage.gateway.accessors import (
    AnalyticsTables,
    CoreTables,
    DocsViews,
    GraphTables,
)
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.factory import (
    build_snapshot_gateway_resolver,
    open_gateway,
    open_memory_gateway,
)
from codeintel.storage.gateway.protocol import (
    DuckDBBinderException,
    DuckDBCatalogException,
    DuckDBConnection,
    DuckDBConnectionException,
    DuckDBDatabaseError,
    DuckDBError,
    DuckDBInvalidInputException,
    DuckDBProgrammingError,
    DuckDBRelation,
    SnapshotGatewayResolver,
    StorageGateway,
)
from codeintel.storage.validation import table_has_rows_for_snapshot

__all__ = [
    "AnalyticsTables",
    "CoreTables",
    "DocsViews",
    "DuckDBBinderException",
    "DuckDBCatalogException",
    "DuckDBConnection",
    "DuckDBConnectionException",
    "DuckDBDatabaseError",
    "DuckDBError",
    "DuckDBInvalidInputException",
    "DuckDBProgrammingError",
    "DuckDBRelation",
    "GraphTables",
    "SnapshotGatewayResolver",
    "StorageConfig",
    "StorageGateway",
    "build_snapshot_gateway_resolver",
    "open_gateway",
    "open_memory_gateway",
    "table_has_rows_for_snapshot",
]
