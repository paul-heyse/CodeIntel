"""Hexagonal architecture port interfaces for the analytics package.

This package defines Protocol-based interfaces that abstract I/O operations,
enabling pure computation to be separated from infrastructure concerns.

Port Categories
---------------
- StoragePort: Re-export from graphs.ports for database operations
- CatalogPort: Re-export from graphs.ports for function catalog access
- GraphRuntimePort: Protocol for graph runtime access

Example
-------
```python
from codeintel.analytics.ports import StoragePort, CatalogPort, GraphRuntimePort


def analyze(storage: StoragePort, catalog: CatalogPort) -> dict[str, int]:
    spans = catalog.function_spans
    # Pure computation on catalog data...
```
"""

from __future__ import annotations

from codeintel.analytics.ports.catalog import CatalogPort, FunctionSpanData
from codeintel.analytics.ports.graphs import GraphRuntimePort
from codeintel.analytics.ports.storage import BatchResult, QueryResult, StoragePort

__all__ = [
    "BatchResult",
    "CatalogPort",
    "FunctionSpanData",
    "GraphRuntimePort",
    "QueryResult",
    "StoragePort",
]
