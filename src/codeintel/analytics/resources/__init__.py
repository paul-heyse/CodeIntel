"""Resource providers for lazy loading analytics resources.

This package provides a lazy-loading resource system with fine-grained,
on-demand resource loading for analytics computations.

Key Components
--------------
ResourceProvider
    Protocol for lazy resource loading.
ResourceRegistry
    Central registry for typed resource access.
CatalogProvider
    Lazy loader for function catalog.
AstProvider
    Lazy loader for parsed AST maps.
FeaturesProvider
    Lazy loader for function AST features.
ModuleMapProvider
    Lazy loader for path-to-module mapping.
ProviderFactory
    Simplified factory for creating and registering providers.

Architecture
------------
Resources are loaded lazily on first access, reducing memory footprint
and startup time. The registry provides type-safe access with clear
error messages for missing resources.

Example
-------
>>> from codeintel.analytics.resources import ProviderFactory
>>> from codeintel.core.resources import ResourceRegistry
>>> factory = ProviderFactory(gateway, snapshot)
>>> options = ProviderRegistryOptions(include_graphs=True, include_catalog=True)
>>> registry = factory.create_registry(ResourceRegistry(), options=options)
>>> from codeintel.graphs.resources.graph_provider import GraphProvider
>>> graphs = registry.require(GraphProvider)
>>> call_graph = graphs.call_graph
"""

from __future__ import annotations

from codeintel.analytics.resources.asts import AstProvider
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.resources.factory import (
    ProviderFactory,
    ProviderFactoryOptions,
    ProviderRegistryOptions,
    build_registry,
)
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.analytics.resources.module_map import ModuleMapProvider
from codeintel.core.resources import (
    ResourceError,
    ResourceNotFoundError,
    ResourceNotLoadedError,
    ResourceProvider,
    ResourceRegistry,
)

__all__ = [
    "AstProvider",
    "CatalogProvider",
    "FeaturesProvider",
    "ModuleMapProvider",
    "ProviderFactory",
    "ProviderFactoryOptions",
    "ProviderRegistryOptions",
    "ResourceError",
    "ResourceNotFoundError",
    "ResourceNotLoadedError",
    "ResourceProvider",
    "ResourceRegistry",
    "build_registry",
]
