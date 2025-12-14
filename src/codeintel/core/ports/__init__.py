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

Example
-------
```python
from codeintel.core.ports import BaseQueryResult, BaseBatchResult

def process_result(result: BaseQueryResult) -> int:
    return result.row_count
```
"""

from __future__ import annotations

from codeintel.core.ports.results import BaseBatchResult, BaseQueryResult

__all__ = [
    "BaseBatchResult",
    "BaseQueryResult",
]

