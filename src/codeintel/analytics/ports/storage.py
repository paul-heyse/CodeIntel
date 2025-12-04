"""Storage port interface re-export for analytics.

This module re-exports the StoragePort protocol from graphs.ports,
providing a consistent import path for analytics modules.
"""

from __future__ import annotations

from codeintel.graphs.ports.storage import BatchResult, QueryResult, StoragePort

__all__ = [
    "BatchResult",
    "QueryResult",
    "StoragePort",
]
