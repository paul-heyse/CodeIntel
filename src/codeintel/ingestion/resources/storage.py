"""Storage provider for lazy access to storage adapters.

This module provides `StorageProvider`, a resource provider that
lazily creates storage adapters for database operations.

NOTE: Imports inside methods are intentional to avoid circular dependencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.ingestion.resources.protocol import LazyResource

if TYPE_CHECKING:
    from codeintel.ingestion.adapters import DuckDBStorageAdapter
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class StorageProvider(LazyResource["DuckDBStorageAdapter"]):
    """Lazy provider for storage adapter.

    Lazily create the DuckDB storage adapter from a gateway.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    """

    def __init__(self, gateway: StorageGateway) -> None:
        """Initialize the storage provider.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        """
        super().__init__("StorageProvider")
        self._gateway = gateway

    def _load(self) -> DuckDBStorageAdapter:
        """Load the storage adapter.

        Returns
        -------
        DuckDBStorageAdapter
            The storage adapter.
        """
        from codeintel.ingestion.adapters import DuckDBStorageAdapter

        log.debug("Creating storage adapter")
        return DuckDBStorageAdapter(self._gateway)

    @property
    def gateway(self) -> StorageGateway:
        """Return the underlying gateway.

        Returns
        -------
        StorageGateway
            The storage gateway.
        """
        return self._gateway


__all__ = ["StorageProvider"]
