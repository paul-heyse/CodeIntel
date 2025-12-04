"""Protocol and base classes for resource providers.

This module re-exports unified resource provider types from codeintel.core.resources,
providing a consistent interface for analytics resource management.

The canonical protocol and base class definitions live in codeintel.core.resources.
This module exists for backward compatibility and to provide a single import point
for analytics code.
"""

from __future__ import annotations

from codeintel.core.resources import (
    LazyResource,
    ResourceError,
    ResourceNotFoundError,
    ResourceNotLoadedError,
    ResourceProvider,
    ResourceProviderBase,
    ResourceRegistry,
)

__all__ = [
    "LazyResource",
    "ResourceError",
    "ResourceNotFoundError",
    "ResourceNotLoadedError",
    "ResourceProvider",
    "ResourceProviderBase",
    "ResourceRegistry",
]
