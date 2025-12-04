"""Storage layer for CodeIntel DuckDB persistence.

This module provides the primary abstractions for database access:

- StorageGateway: Protocol for DuckDB access with dataset registry
- StorageConfig: Configuration for opening gateways
- DatasetRegistry: In-memory view of registered datasets
- DuckDBConnection: Type alias for the underlying connection

Due to module initialization order constraints, imports should be done
from specific submodules:

    from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway
    from codeintel.storage.datasets import DatasetRegistry, load_dataset_registry
    from codeintel.storage.repositories import fetch_models, DataModelRow
"""

from __future__ import annotations

# Note: We avoid importing from gateway here to prevent circular imports.
# Users should import StorageConfig from codeintel.storage.gateway instead.

__all__: list[str] = []
