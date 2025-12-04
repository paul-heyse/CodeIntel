"""Central registry for typed resource access.

This module re-exports the unified ResourceRegistry from codeintel.core.resources,
providing a consistent interface for analytics resource management.

The canonical implementation lives in codeintel.core.resources.registry.
"""

from __future__ import annotations

from codeintel.analytics.resources.protocol import ResourceError
from codeintel.core.resources.registry import ResourceNotFoundError, ResourceRegistry

__all__ = [
    "ResourceError",
    "ResourceNotFoundError",
    "ResourceRegistry",
]
