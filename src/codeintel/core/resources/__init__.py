"""Unified resource provider infrastructure for graphs and analytics.

This package provides a single resource provider protocol used by both
the graphs and analytics subsystems, eliminating protocol duplication.

Key Components
--------------
ResourceProvider
    Protocol for lazy resource loading.
ResourceProviderBase
    Simple base class with caching.
LazyResource
    Extended base class with error tracking, optional access, and DI support.
ResourceRegistry
    Central registry for typed resource access.
ResourceError
    Base exception for resource-related errors.
ResourceNotFoundError
    Raised when a required resource is not registered.
ResourceNotLoadedError
    Raised when a lazy resource fails to load.
"""

from __future__ import annotations

from codeintel.core.resources.protocol import (
    LazyResource,
    ResourceError,
    ResourceNotLoadedError,
    ResourceProvider,
    ResourceProviderBase,
)
from codeintel.core.resources.registry import (
    ResourceNotFoundError,
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
