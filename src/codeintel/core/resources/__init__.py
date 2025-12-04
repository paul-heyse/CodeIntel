"""Unified resource provider infrastructure for graphs and analytics.

This package provides a single resource provider protocol used by both
the graphs and analytics subsystems, eliminating protocol duplication.

Modules
-------
- protocol: Unified resource provider protocol
- registry: Unified resource registry
"""

from __future__ import annotations

from codeintel.core.resources.protocol import ResourceProvider, ResourceProviderBase
from codeintel.core.resources.registry import ResourceNotFoundError, ResourceRegistry

__all__ = [
    "ResourceNotFoundError",
    "ResourceProvider",
    "ResourceProviderBase",
    "ResourceRegistry",
]
