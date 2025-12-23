"""Storage layer for CodeIntel DuckDB persistence.

This module provides the primary abstractions for database access:

- StorageGateway: Protocol for DuckDB access with dataset registry
- StorageConfig: Configuration for opening gateways
- DatasetRegistry: In-memory view of registered datasets
- DuckDBConnection: Type alias for the underlying connection
- DuckDBPolicyBackend: Centralized DDL and mutation operations

Due to circular import constraints, most submodule symbols are NOT re-exported
at the package level. Import directly from submodules for most use cases.

Recommended import patterns::

    from codeintel.storage.gateway import StorageConfig, StorageGateway
    from codeintel.storage.gateway import open_gateway
    from codeintel.storage.datasets import DatasetRegistry, load_dataset_registry
    from codeintel.storage.repositories import fetch_models, DataModelRow
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

from typing import TYPE_CHECKING

from codeintel.core.imports.lazy import lazy_import

__all__ = ["StorageFacade"]

if TYPE_CHECKING:
    from codeintel.storage.facade import StorageFacade

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "StorageFacade": ("codeintel.storage.facade", "StorageFacade"),
}


def __getattr__(name: str) -> object:
    """Lazily import storage symbols to avoid import-time cycles.

    Returns
    -------
    object
        Requested attribute loaded from its defining module.

    Raises
    ------
    AttributeError
        If the requested attribute is not registered for lazy loading.
    """
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = lazy_import(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
