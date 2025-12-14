"""Catalog resource provider for function catalog access.

This module provides `CatalogProvider` for lazy loading of the function
catalog used in analytics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.core.resources import LazyResource, ResourceNotLoadedError
from codeintel.graphs.catalog import CatalogService

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.graphs.catalog import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class CatalogProvider(LazyResource["FunctionCatalogProvider"]):
    """Provider for function catalog with lazy loading.

    The function catalog contains metadata about all functions in the
    codebase including their relationships and attributes.

    Parameters can be None when using factory methods like `from_catalog()`
    that set a pre-loaded resource.

    Example
    -------
    >>> provider = CatalogProvider(gateway, snapshot)
    >>> catalog = provider.get()
    >>> function_info = catalog.get_function(function_goid)
    """

    RESOURCE_NAME: ClassVar[str] = "FunctionCatalog"

    def __init__(
        self,
        gateway: StorageGateway | None = None,
        snapshot: SnapshotRef | None = None,
    ) -> None:
        """Initialize the catalog provider.

        Parameters
        ----------
        gateway
            Storage gateway for database access. Can be None if using
            `set_preloaded()` or `from_catalog()` factory method.
        snapshot
            Repository snapshot reference. Can be None if using
            `set_preloaded()` or `from_catalog()` factory method.
        """
        super().__init__("FunctionCatalog")
        self._gateway = gateway
        self._snapshot = snapshot

    @classmethod
    def from_catalog(cls, catalog: FunctionCatalogProvider) -> CatalogProvider:
        """Create a provider from an existing catalog.

        Use this factory when a catalog has already been loaded and you
        want to wrap it in a provider for the resource registry.

        Parameters
        ----------
        catalog
            Pre-loaded function catalog provider.

        Returns
        -------
        CatalogProvider
            Provider wrapping the existing catalog.

        Example
        -------
        >>> existing_catalog = CatalogService.from_db(gateway, repo=repo, commit=commit)
        >>> provider = CatalogProvider.from_catalog(existing_catalog)
        >>> registry.register(CatalogProvider, provider)
        """
        provider = cls(gateway=None, snapshot=None)
        provider.set_preloaded(catalog)
        return provider

    def _load(self) -> FunctionCatalogProvider:
        """Load the function catalog.

        Returns
        -------
        FunctionCatalogProvider
            The loaded catalog.

        Raises
        ------
        ResourceNotLoadedError
            If gateway or snapshot are None (provider created for pre-loading only).
        """
        if self._gateway is None or self._snapshot is None:
            raise ResourceNotLoadedError(
                self._name,
                "Cannot load - provider was created for pre-loaded resource only. "
                "Use from_catalog() with a pre-loaded catalog or provide gateway and snapshot.",
            )

        return _load_function_catalog(self._gateway, self._snapshot)


def _load_function_catalog(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> FunctionCatalogProvider:
    return CatalogService.from_db(
        gateway,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )


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

        Default implementation returns empty dict. Subclasses override
        with specific queries.

        Returns
        -------
        dict[int, object]
            Mapping of GOID to catalog data.
        """
        _ = self._gateway.con
        return {}


__all__ = [
    "CatalogProvider",
    "CatalogQueryProvider",
]
