"""Gateway package for DuckDB storage access.

This package provides the StorageGateway protocol and related types for
accessing CodeIntel DuckDB databases.

Architecture
------------
The gateway module uses a layered accessor pattern:

- **BaseTableAccessor**: Base class providing `_table()` and `con` access
- **CoreTables**, **GraphTables**, **AnalyticsTables**: Schema-specific accessor classes
  that inherit from BaseTableAccessor and provide typed methods for each table/view
- **DocsViews**: Read-only accessor for docs.* views
- **DuckDBGateway**: Concrete implementation combining all accessors

Usage
-----
Open a gateway and access tables through typed accessors:

    from codeintel.storage.gateway import open_gateway, StorageConfig

    config = StorageConfig(path="catalog.duckdb")
    with open_gateway(config) as gw:

        modules = gw.core.modules().fetchall()

Writes are routed through `codeintel.storage.warehouse.Warehouse` to keep a single
I/O boundary.

The accessor classes provide:
- **Relation methods**: Return DuckDBRelation for query building
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.storage.exceptions import QueryError as StorageQueryError
    from codeintel.storage.exceptions import SchemaError as StorageSchemaError
    from codeintel.storage.exceptions import StorageConnectionError, StorageError
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
    from codeintel.storage.gateway.minimal import MinimalStorageGateway
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
    from codeintel.storage.ibis_adapter import IbisGateway
    from codeintel.storage.validation import table_has_rows_for_snapshot


_EXPORTS: dict[str, tuple[str, str]] = {
    "AnalyticsTables": ("codeintel.storage.gateway.accessors", "AnalyticsTables"),
    "BaseTableAccessor": ("codeintel.storage.gateway.accessors", "BaseTableAccessor"),
    "CoreTables": ("codeintel.storage.gateway.accessors", "CoreTables"),
    "DocsViews": ("codeintel.storage.gateway.accessors", "DocsViews"),
    "DuckDBBinderException": ("codeintel.storage.gateway.protocol", "DuckDBBinderException"),
    "DuckDBCatalogException": ("codeintel.storage.gateway.protocol", "DuckDBCatalogException"),
    "DuckDBConnection": ("codeintel.storage.gateway.protocol", "DuckDBConnection"),
    "DuckDBConnectionException": (
        "codeintel.storage.gateway.protocol",
        "DuckDBConnectionException",
    ),
    "DuckDBDatabaseError": ("codeintel.storage.gateway.protocol", "DuckDBDatabaseError"),
    "DuckDBError": ("codeintel.storage.gateway.protocol", "DuckDBError"),
    "DuckDBInvalidInputException": (
        "codeintel.storage.gateway.protocol",
        "DuckDBInvalidInputException",
    ),
    "DuckDBProgrammingError": ("codeintel.storage.gateway.protocol", "DuckDBProgrammingError"),
    "DuckDBRelation": ("codeintel.storage.gateway.protocol", "DuckDBRelation"),
    "GraphTables": ("codeintel.storage.gateway.accessors", "GraphTables"),
    "IbisGateway": ("codeintel.storage.ibis_adapter", "IbisGateway"),
    "MinimalStorageGateway": ("codeintel.storage.gateway.minimal", "MinimalStorageGateway"),
    "SnapshotGatewayResolver": ("codeintel.storage.gateway.protocol", "SnapshotGatewayResolver"),
    "StorageConfig": ("codeintel.storage.gateway.config", "StorageConfig"),
    "StorageConnectionError": ("codeintel.storage.exceptions", "StorageConnectionError"),
    "StorageError": ("codeintel.storage.exceptions", "StorageError"),
    "StorageGateway": ("codeintel.storage.gateway.protocol", "StorageGateway"),
    "StorageQueryError": ("codeintel.storage.exceptions", "QueryError"),
    "StorageSchemaError": ("codeintel.storage.exceptions", "SchemaError"),
    "build_snapshot_gateway_resolver": (
        "codeintel.storage.gateway.factory",
        "build_snapshot_gateway_resolver",
    ),
    "open_gateway": ("codeintel.storage.gateway.factory", "open_gateway"),
    "open_memory_gateway": ("codeintel.storage.gateway.factory", "open_memory_gateway"),
    "table_has_rows_for_snapshot": (
        "codeintel.storage.validation",
        "table_has_rows_for_snapshot",
    ),
}


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


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
    "IbisGateway",
    "MinimalStorageGateway",
    "SnapshotGatewayResolver",
    "StorageConfig",
    "StorageConnectionError",
    "StorageError",
    "StorageGateway",
    "StorageQueryError",
    "StorageSchemaError",
    "build_snapshot_gateway_resolver",
    "open_gateway",
    "open_memory_gateway",
    "table_has_rows_for_snapshot",
]
