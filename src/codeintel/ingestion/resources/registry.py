"""Central registry for typed resource access in ingestion.

This module re-exports from codeintel.core.resources for consistency
with the unified resource infrastructure used across graphs and analytics.

The core registry provides:
- ResourceRegistry: Central registry for typed resource providers
- ResourceNotFoundError: Exception for missing resources
"""

from __future__ import annotations

from codeintel.core.resources import (
    ResourceNotFoundError,
    ResourceRegistry,
)

__all__ = [
    "ResourceNotFoundError",
    "ResourceRegistry",
]
