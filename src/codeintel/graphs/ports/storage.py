"""Storage data types for database operations.

This module re-exports unified storage types from ``codeintel.core.ports.storage``.

Data Classes
------------
- QueryResult: Result of a database query operation
- BatchResult: Result of a batch insert/update operation

See Also
--------
codeintel.core.ports.storage : Canonical storage types
codeintel.graphs.resources.storage : StorageResource for storage access
"""

from __future__ import annotations

from codeintel.core.ports.storage import (
    BatchResult,
    MutableQueryResult,
    QueryResult,
)

__all__ = [
    "BatchResult",
    "MutableQueryResult",
    "QueryResult",
]
