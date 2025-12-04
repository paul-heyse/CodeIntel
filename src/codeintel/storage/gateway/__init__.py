"""Gateway package for DuckDB storage access.

This package provides the StorageGateway protocol and related types for
accessing CodeIntel DuckDB databases.

Architecture
------------
The gateway module uses a layered accessor pattern:

- **BaseTableAccessor**: Base class providing `_table()` and `_insert_rows()` methods
- **CoreTables**, **GraphTables**, **AnalyticsTables**: Schema-specific accessor classes
  that inherit from BaseTableAccessor and provide typed methods for each table
- **DocsViews**: Read-only accessor for docs.* views
- **DuckDBGateway**: Concrete implementation combining all accessors

Usage
-----
Open a gateway and access tables through typed accessors:

    from codeintel.storage.gateway import open_gateway, StorageConfig

    config = StorageConfig(path="catalog.duckdb")
    with open_gateway(config) as gw:
        # Typed relation access
        modules = gw.core.modules().fetchall()

        # Typed row insertion
        gw.core.insert_goids([(hash, urn, repo, commit, ...)])

The accessor classes provide:
- **Relation methods**: Return DuckDBRelation for query building
- **Insert methods**: Typed bulk insertion with schema validation
"""

from __future__ import annotations

from codeintel.storage.gateway.accessors import (
    AnalyticsTables,
    BaseTableAccessor,
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
    "BaseTableAccessor",
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
