"""Storage layer for CodeIntel DuckDB persistence.

This module provides the primary abstractions for database access:

- StorageGateway: Protocol for DuckDB access with dataset registry
- StorageConfig: Configuration for opening gateways
- DatasetRegistry: In-memory view of registered datasets
- DuckDBConnection: Type alias for the underlying connection
- GatewayCache: Thread-safe gateway caching for connection reuse
- DuckDBPolicyBackend: Centralized DDL and mutation operations

Due to circular import constraints, most submodule symbols are NOT re-exported
at the package level. Import directly from submodules for most use cases.

Recommended import patterns::

    from codeintel.storage.gateway import StorageConfig, StorageGateway
    from codeintel.storage.gateway import open_gateway
    from codeintel.storage.datasets import DatasetRegistry, load_dataset_registry
    from codeintel.storage.repositories import fetch_models, DataModelRow
    from codeintel.storage.gateway_cache import get_gateway, close_gateways
    from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

Circular Import Note
--------------------
Some config/dataset modules import storage view builders. To avoid introducing
import cycles, this package intentionally keeps its public surface small and
does not re-export most gateway/view helpers.

Import DuckDBPolicyBackend directly from its submodule to avoid circular
import issues when storage.views depends on storage.gateway.
"""

from __future__ import annotations

__all__: list[str] = []
