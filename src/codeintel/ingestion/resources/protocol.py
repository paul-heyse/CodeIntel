"""Protocol and base classes for ingestion resource providers.

This module re-exports from codeintel.core.resources for consistency
with the unified resource infrastructure used across graphs and analytics.

The core resource types provide:
- LazyResource: ABC with caching, error tracking, get_or_none(), set_preloaded()
- ResourceProvider: Protocol for lazy resource loading
- ResourceProviderBase: Simple base class with caching
- ResourceError: Base exception for resource-related errors
- ResourceNotLoadedError: Exception for lazy resource load failures
"""

from __future__ import annotations

from codeintel.core.resources import (
    LazyResource,
    ResourceError,
    ResourceNotLoadedError,
    ResourceProvider,
    ResourceProviderBase,
)

__all__ = [
    "LazyResource",
    "ResourceError",
    "ResourceNotLoadedError",
    "ResourceProvider",
    "ResourceProviderBase",
]
