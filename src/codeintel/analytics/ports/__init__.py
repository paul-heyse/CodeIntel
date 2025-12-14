"""Hexagonal architecture port interfaces for the analytics package.

This package re-exports types from ``codeintel.graphs.ports`` for convenience.
New code should prefer importing from graphs directly.

Port Categories
---------------
- StoragePort: Database query and batch operations (deprecated)
- CatalogPort: Function catalog access (deprecated)

Data Types
----------
- FunctionSpan: Unified function span representation
- BatchResult, QueryResult: Storage operation results

.. deprecated:: 5.0.0
    Import directly from ``codeintel.graphs.ports`` or use the resource classes
    from ``codeintel.graphs.resources`` instead.

Example
-------
```python
from codeintel.graphs.catalog import CatalogService
from codeintel.graphs.resources import StorageResource


def analyze(storage: StorageResource, catalog: CatalogService) -> dict[str, int]:
    spans = catalog.function_spans
```
"""

from __future__ import annotations

from codeintel.graphs.catalog import CatalogService, FunctionSpan
from codeintel.graphs.ports import (
    BatchResult,
    CatalogPort,
    FunctionSpanData,
    QueryResult,
    StoragePort,
)

__all__ = [
    "BatchResult",
    "CatalogPort",  # Deprecated
    "CatalogService",  # Canonical
    "FunctionSpan",
    "FunctionSpanData",  # Deprecated alias
    "QueryResult",
    "StoragePort",  # Deprecated
]
