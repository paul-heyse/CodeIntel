"""Catalog resource provider for function catalog access.

This module provides `CatalogProvider` for lazy loading of the function
catalog used in analytics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.analytics.resources.protocol import LazyResource

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.graphs.function_catalog_provider import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class CatalogProvider(LazyResource["FunctionCatalogProvider"]):
    """Provider for function catalog with lazy loading.

    The function catalog contains metadata about all functions in the
    codebase including their relationships and attributes.

    Example
    -------
    >>> provider = CatalogProvider(gateway, snapshot)
    >>> catalog = provider.get()
    >>> function_info = catalog.get_function(function_goid)
    """

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Initialize the catalog provider.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        """
        super().__init__("FunctionCatalog")
        self._gateway = gateway
        self._snapshot = snapshot

    def _load(self) -> FunctionCatalogProvider:
        """Load the function catalog.

        Returns
        -------
        FunctionCatalogProvider
            The loaded catalog.
        """
        from codeintel.graphs.function_catalog_service import FunctionCatalogService

        service = FunctionCatalogService(self._gateway)
        return service.get_provider(self._snapshot.repo, self._snapshot.commit)


class CatalogQueryProvider(LazyResource[dict[int, object]]):
    """Provider for catalog query results.

    Use this for pre-filtered or aggregated catalog data that's used
    by multiple plugins.
    """

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        *,
        query_name: str = "default",
    ) -> None:
        """Initialize the query provider.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        query_name
            Name identifier for the query (for caching).
        """
        super().__init__(f"CatalogQuery:{query_name}")
        self._gateway = gateway
        self._snapshot = snapshot
        self._query_name = query_name

    def _load(self) -> dict[int, object]:
        """Load query results.

        Returns
        -------
        dict[int, object]
            Mapping of GOID to catalog data.
        """
        # Default implementation returns empty dict
        # Subclasses override with specific queries
        return {}


__all__ = [
    "CatalogProvider",
    "CatalogQueryProvider",
]
