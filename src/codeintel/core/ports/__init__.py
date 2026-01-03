"""Shared port data types for cross-package use.

This package provides base data types used across analytics, graphs,
and ingestion packages. It establishes a common interface for result
types returned by port operations.

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

from codeintel.core.ports.export import (
    ExportRelation,
    RecordBatch,
    RecordBatchReader,
    ResultStream,
)
from codeintel.core.ports.storage import (
    BatchResult,
    MutableQueryResult,
    QueryResult,
    StoragePort,
)

__all__ = [
    "BatchResult",
    "ExportRelation",
    "MutableQueryResult",
    "QueryResult",
    "RecordBatch",
    "RecordBatchReader",
    "ResultStream",
    "StoragePort",
]
