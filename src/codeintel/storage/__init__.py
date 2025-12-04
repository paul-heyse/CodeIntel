"""Storage layer for CodeIntel DuckDB persistence.

This module provides the primary abstractions for database access:

- StorageGateway: Protocol for DuckDB access with dataset registry
- StorageConfig: Configuration for opening gateways
- DatasetRegistry: In-memory view of registered datasets
- DuckDBConnection: Type alias for the underlying connection

Due to module initialization order constraints, imports should be done
from specific submodules:

    from codeintel.storage.config import StorageConfig
    from codeintel.storage.gateway import StorageGateway, open_gateway, DuckDBConnection
    from codeintel.storage.datasets import DatasetRegistry, load_dataset_registry
"""

from __future__ import annotations

from codeintel.storage.config import StorageConfig

__all__ = [
    "StorageConfig",
]
