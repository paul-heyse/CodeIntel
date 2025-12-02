"""Resource providers for lazy loading analytics resources.

This package provides a lazy-loading resource system that replaces the
monolithic `AnalyticsContext` with fine-grained, on-demand resource
loading.

Key Components
--------------
ResourceProvider
    Protocol for lazy resource loading.
ResourceRegistry
    Central registry for typed resource access.
GraphProvider
    Lazy loader for graph resources (call, import, symbol graphs).
CatalogProvider
    Lazy loader for function catalog.
AstProvider
    Lazy loader for parsed AST maps.
AnalyticsContextProvider
    Lazy loader for legacy AnalyticsContext (for backward compatibility).

Architecture
------------
Resources are loaded lazily on first access, reducing memory footprint
and startup time. The registry provides type-safe access with clear
error messages for missing resources.

Example
-------
>>> from codeintel.analytics.resources import ResourceRegistry, GraphProvider
>>> registry = ResourceRegistry()
>>> registry.register(GraphProvider, GraphProvider(gateway, snapshot))
>>> graphs = registry.get(GraphProvider)
>>> call_graph = graphs.call_graph  # Loaded on first access
"""

from __future__ import annotations

from codeintel.analytics.resources.analytics_context import AnalyticsContextProvider
from codeintel.analytics.resources.asts import AstProvider
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.resources.graphs import GraphProvider
from codeintel.analytics.resources.protocol import (
    ResourceError,
    ResourceNotLoadedError,
    ResourceProvider,
)
from codeintel.analytics.resources.registry import (
    ResourceNotFoundError,
    ResourceRegistry,
)

__all__ = [
    "AnalyticsContextProvider",
    "AstProvider",
    "CatalogProvider",
    "GraphProvider",
    "ResourceError",
    "ResourceNotFoundError",
    "ResourceNotLoadedError",
    "ResourceProvider",
    "ResourceRegistry",
]
