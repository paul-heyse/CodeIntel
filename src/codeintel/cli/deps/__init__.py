"""Dependency injection for CLI commands.

This package provides the dependency container and protocols used by the
new Command[T] pattern for explicit, testable dependency management.

Public API
----------
- ``Deps``: Main dependency container commands receive
- ``DepsBuilder``: Builder for constructing Deps instances
- ``StorageAccess``: Protocol for storage operations
- ``JobManagerProtocol``: Protocol for job management
- ``ServingAccess``: Protocol for serving layer
- ``LazyStorageProvider``: Lazy-loading storage implementation
- ``LazyServingProvider``: Lazy-loading serving implementation

Compatibility
-------------
- ``deps_from_handler_context``: Create Deps from legacy HandlerContext
- ``handler_context_from_deps``: Create HandlerContext from Deps
- ``wrap_legacy_handler``: Wrap legacy handler for new pattern
"""

from __future__ import annotations

from codeintel.cli.deps.compat import (
    deps_from_handler_context,
    handler_context_from_deps,
    wrap_legacy_handler,
)
from codeintel.cli.deps.container import Deps, DepsBuilder
from codeintel.cli.deps.protocols import JobManagerProtocol, ServingAccess, StorageAccess
from codeintel.cli.deps.providers import LazyServingProvider, LazyStorageProvider

__all__ = [
    "Deps",
    "DepsBuilder",
    "JobManagerProtocol",
    "LazyServingProvider",
    "LazyStorageProvider",
    "ServingAccess",
    "StorageAccess",
    "deps_from_handler_context",
    "handler_context_from_deps",
    "wrap_legacy_handler",
]
