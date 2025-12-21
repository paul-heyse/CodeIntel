"""Provider factory for simplified resource provider creation.

This module provides `ProviderFactory` which simplifies the creation and
registration of resource providers for analytics contexts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.analytics.resources.asts import AstProvider
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.analytics.resources.module_map import ModuleMapProvider
from codeintel.core.resources import ResourceRegistry
from codeintel.graphs.resources.graph_provider import GraphProvider
from codeintel.graphs.runtime import GraphRuntimeOptions

if TYPE_CHECKING:
    from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
    from codeintel.core.catalog import FunctionCatalogProvider
    from codeintel.graphs.resources.graph_provider import GraphRuntimeLike
    from codeintel.graphs.runtime import GraphRuntime
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass
class ProviderFactoryOptions:
    """Options for configuring provider creation.

    Attributes
    ----------
    graph_backend
        Backend configuration for graphs (CPU/GPU).
    graph_options
        Full graph runtime options (overrides graph_backend).
    max_functions
        Maximum number of functions to parse for AST/features.
    language
        Language filter for module map.
    """

    graph_backend: GraphBackendConfig | None = None
    graph_options: GraphRuntimeOptions | None = None
    max_functions: int | None = None
    language: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderRegistryOptions:
    """Options for selecting providers to register.

    Parameters
    ----------
    include_graphs
        Include GraphProvider in the registry.
    include_catalog
        Include CatalogProvider in the registry.
    include_asts
        Include AstProvider in the registry.
    include_features
        Include FeaturesProvider in the registry.
    include_module_map
        Include ModuleMapProvider in the registry.
    """

    include_graphs: bool = True
    include_catalog: bool = True
    include_asts: bool = False
    include_features: bool = False
    include_module_map: bool = False


class ProviderFactory:
    """Factory for creating and registering resource providers.

    This factory simplifies the creation of resource providers by providing
    a single entry point that handles gateway/snapshot configuration.

    Example
    -------
    >>> factory = ProviderFactory(gateway, snapshot)
    >>> registry = ResourceRegistry()
    >>> options = ProviderRegistryOptions(include_graphs=True, include_asts=True)
    >>> registry = factory.create_registry(registry, options=options)
    >>> provider = registry.require(GraphProvider)

    Notes
    -----
    The factory creates providers lazily - resources are only loaded when
    first accessed through the registry.
    """

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        *,
        options: ProviderFactoryOptions | None = None,
    ) -> None:
        """Initialize the factory.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        options
            Optional configuration for provider creation.
        """
        self._gateway = gateway
        self._snapshot = snapshot
        self._options = options or ProviderFactoryOptions()
        self._cached_catalog: CatalogProvider | None = None
        self._cached_graphs: GraphProvider | None = None

    @property
    def gateway(self) -> StorageGateway:
        """Return the storage gateway."""
        return self._gateway

    @property
    def snapshot(self) -> SnapshotRef:
        """Return the snapshot reference."""
        return self._snapshot

    @property
    def options(self) -> ProviderFactoryOptions:
        """Return the factory options."""
        return self._options

    def create_registry(
        self,
        registry: ResourceRegistry,
        *,
        options: ProviderRegistryOptions | None = None,
    ) -> ResourceRegistry:
        """Create a registry with the requested providers.

        Parameters
        ----------
        registry
            Registry instance to populate.
        options
            Provider selection options.

        Returns
        -------
        ResourceRegistry
            Registry populated with the requested providers.

        Example
        -------
        >>> registry = ResourceRegistry()
        >>> options = ProviderRegistryOptions(include_graphs=True, include_asts=True)
        >>> registry = factory.create_registry(registry, options=options)
        """
        resolved = options or ProviderRegistryOptions()
        if resolved.include_catalog:
            registry.register(CatalogProvider, self.make_catalog_provider())

        if resolved.include_graphs:
            registry.register(GraphProvider, self.make_graph_provider())

        if resolved.include_asts:
            registry.register(AstProvider, self.make_ast_provider())

        if resolved.include_features:
            registry.register(FeaturesProvider, self.make_features_provider())

        if resolved.include_module_map:
            registry.register(ModuleMapProvider, self.make_module_map_provider())

        return registry

    def make_catalog_provider(
        self,
        *,
        catalog: FunctionCatalogProvider | None = None,
    ) -> CatalogProvider:
        """Create a catalog provider.

        Parameters
        ----------
        catalog
            Optional pre-loaded catalog to wrap.

        Returns
        -------
        CatalogProvider
            Configured catalog provider.
        """
        if catalog is not None:
            return CatalogProvider.from_catalog(catalog)
        if self._cached_catalog is not None:
            return self._cached_catalog
        provider = CatalogProvider(self._gateway, self._snapshot)
        self._cached_catalog = provider
        return provider

    def make_graph_provider(
        self,
        *,
        runtime: GraphRuntime | GraphRuntimeLike | None = None,
        options: GraphRuntimeOptions | None = None,
    ) -> GraphProvider:
        """Create a graph provider.

        Parameters
        ----------
        runtime
            Optional pre-built runtime to wrap. Can be a GraphRuntime or
            any object implementing GraphRuntimeLike.
        options
            Optional runtime options (overrides factory options).

        Returns
        -------
        GraphProvider
            Configured graph provider.
        """
        if runtime is not None:
            return GraphProvider.from_runtime(runtime)

        if self._cached_graphs is not None:
            return self._cached_graphs

        resolved_options = options or self._options.graph_options
        if resolved_options is None and self._options.graph_backend is not None:
            resolved_options = GraphRuntimeOptions(
                snapshot=self._snapshot,
                backend=self._options.graph_backend,
            )

        provider = GraphProvider.from_gateway(
            self._gateway,
            self._snapshot,
            options=resolved_options,
        )
        self._cached_graphs = provider
        return provider

    def make_ast_provider(
        self,
        *,
        catalog_provider: FunctionCatalogProvider | None = None,
        max_functions: int | None = None,
    ) -> AstProvider:
        """Create an AST provider.

        Parameters
        ----------
        catalog_provider
            Optional catalog provider for GOID resolution.
        max_functions
            Maximum functions to parse (overrides factory options).

        Returns
        -------
        AstProvider
            Configured AST provider.
        """
        resolved_max = max_functions or self._options.max_functions
        return AstProvider(
            gateway=self._gateway,
            snapshot=self._snapshot,
            catalog_provider=catalog_provider,
            max_functions=resolved_max,
        )

    def make_features_provider(
        self,
        *,
        catalog_provider: FunctionCatalogProvider | None = None,
        max_functions: int | None = None,
    ) -> FeaturesProvider:
        """Create a features provider.

        Parameters
        ----------
        catalog_provider
            Optional catalog provider for GOID resolution.
        max_functions
            Maximum functions to process (overrides factory options).

        Returns
        -------
        FeaturesProvider
            Configured features provider.
        """
        resolved_max = max_functions or self._options.max_functions
        return FeaturesProvider(
            gateway=self._gateway,
            snapshot=self._snapshot,
            catalog_provider=catalog_provider,
            max_functions=resolved_max,
        )

    def make_module_map_provider(
        self,
        *,
        language: str | None = None,
    ) -> ModuleMapProvider:
        """Create a module map provider.

        Parameters
        ----------
        language
            Optional language filter (overrides factory options).

        Returns
        -------
        ModuleMapProvider
            Configured module map provider.
        """
        resolved_language = language or self._options.language
        return ModuleMapProvider(
            gateway=self._gateway,
            snapshot=self._snapshot,
            language=resolved_language,
        )

    def clear_cache(self) -> None:
        """Clear cached providers.

        Call this to force new providers to be created on next access.
        """
        self._cached_catalog = None
        self._cached_graphs = None


__all__ = [
    "ProviderFactory",
    "ProviderFactoryOptions",
]
