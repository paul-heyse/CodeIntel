"""Plugin registry infrastructure.

This package contains base classes and utilities for plugin registries:

Base Classes
------------
- BasePluginRegistry: Abstract base class for domain-specific registries
- RegistrablePlugin: Protocol for plugins that can be registered

Plan Types
----------
- PluginPlan: Resolved execution plan with plugins and skips
- PluginSkip: Metadata for skipped plugins

Utilities
---------
- topological_sort: Sort plugins by dependency order
- build_provider_index, build_provider_index_from_metadata: Build lookup maps
"""

from __future__ import annotations

from codeintel.core.plugins.registry.base import (
    BasePluginRegistry,
    DefaultRegistryHooks,
    PluginPlan,
    PluginSkip,
    RegistrablePlugin,
    RegistryEntry,
    RegistryHooks,
)
from codeintel.core.plugins.registry.sorting import (
    build_provider_index,
    build_provider_index_from_metadata,
    topological_sort,
)

__all__ = [
    "BasePluginRegistry",
    "DefaultRegistryHooks",
    "PluginPlan",
    "PluginSkip",
    "RegistrablePlugin",
    "RegistryEntry",
    "RegistryHooks",
    "build_provider_index",
    "build_provider_index_from_metadata",
    "topological_sort",
]
