"""Resource providers for lazy loading ingestion resources.

This package provides a lazy-loading resource system that enables
fine-grained, on-demand resource loading for ingestion plugins.

Key Components
--------------
ResourceProvider
    Protocol for lazy resource loading.
ResourceRegistry
    Central registry for typed resource access.
ModuleProvider
    Lazy loader for module records (from tracker or inventory).
TrackerProvider
    Lazy loader for change tracker.
ToolsProvider
    Lazy loader for tool service.

Architecture
------------
Resources are loaded lazily on first access, reducing memory footprint
and startup time. The registry provides type-safe access with clear
error messages for missing resources.

Example
-------
>>> from codeintel.ingestion.resources import ResourceRegistry, ModuleProvider
>>> registry = ResourceRegistry()
>>> registry.register(ModuleProvider, ModuleProvider(gateway, snapshot))
>>> modules = registry.require(ModuleProvider)  # Loaded on first access
"""

from __future__ import annotations

from codeintel.ingestion.resources.modules import ModuleProvider
from codeintel.ingestion.resources.protocol import (
    LazyResource,
    ResourceError,
    ResourceNotLoadedError,
    ResourceProvider,
)
from codeintel.ingestion.resources.registry import (
    ResourceNotFoundError,
    ResourceRegistry,
)
from codeintel.ingestion.resources.tools import ToolsProvider
from codeintel.ingestion.resources.tracker import TrackerConfig, TrackerProvider

__all__ = [
    "LazyResource",
    "ModuleProvider",
    "ResourceError",
    "ResourceNotFoundError",
    "ResourceNotLoadedError",
    "ResourceProvider",
    "ResourceRegistry",
    "ToolsProvider",
    "TrackerConfig",
    "TrackerProvider",
]
