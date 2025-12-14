"""Storage port interface for database operations.

This module re-exports unified storage types from ``codeintel.core.ports.storage``.

Data Classes
------------
- QueryResult: Result of a database query operation
- BatchResult: Result of a batch insert/update operation

Deprecated
----------
- StoragePort: Use StorageResource directly instead

.. deprecated:: 5.0.0
    The StoragePort protocol is deprecated. Use StorageResource from
    ``codeintel.graphs.resources.storage`` directly instead.

See Also
--------
codeintel.core.ports.storage : Canonical storage types
codeintel.core.ports.BaseQueryResult : Base protocol for query results
codeintel.core.ports.BaseBatchResult : Base protocol for batch results
"""

from __future__ import annotations

# Re-export unified types from core for backward compatibility
from codeintel.core.ports.storage import (
    BatchResult,
    MutableQueryResult,
    QueryResult,
    StoragePort,
)

__all__ = [
    "BatchResult",
    "MutableQueryResult",
    "QueryResult",
    "StoragePort",
]
