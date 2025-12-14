"""Shared port data types for cross-package use.

This package provides base data types used across analytics, graphs,
and ingestion packages. It establishes a common interface for result
types returned by port operations.

Base Types
----------
BaseQueryResult
    Protocol for query result types with rows and row_count.
BaseBatchResult
    Protocol for batch operation result types.

Unified Types
-------------
QueryResult
    Unified query result for all storage operations.
BatchResult
    Unified batch operation result.
StoragePort
    Protocol for storage access operations.

Example
-------
```python
from codeintel.core.ports import QueryResult, BatchResult, StoragePort


def handle_result(result: QueryResult) -> int:
    return result.row_count
```
"""

from __future__ import annotations

from codeintel.core.ports.results import BaseBatchResult, BaseQueryResult
from codeintel.core.ports.storage import (
    BatchResult,
    MutableQueryResult,
    QueryResult,
    StoragePort,
)

__all__ = [
    "BaseBatchResult",
    "BaseQueryResult",
    "BatchResult",
    "MutableQueryResult",
    "QueryResult",
    "StoragePort",
]
