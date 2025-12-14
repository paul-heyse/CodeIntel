"""Data transfer objects (DTOs) for the graphs package.

This package defines lightweight data types for transferring data
between layers without coupling to specific implementations.

Data Types
----------
- BatchResult, QueryResult: Storage operation results
- GraphData: Lightweight graph data transfer object
- ParsedFunction, ParsedModule: Parsed source code representations
- FunctionSpan: Unified function span (via re-export from catalog)

Usage
-----
For dependency injection, use the resource classes from ``graphs.resources``
or service classes from ``graphs.catalog`` directly.

Example
-------
```python
from codeintel.graphs.resources import StorageResource
from codeintel.graphs.catalog import CatalogService


def process(storage: StorageResource, catalog: CatalogService) -> None: ...
```
"""

from __future__ import annotations

from codeintel.graphs.catalog import FunctionSpan
from codeintel.graphs.ports.engine import GraphData
from codeintel.graphs.ports.parsing import (
    ParsedFunction,
    ParsedModule,
)
from codeintel.graphs.ports.storage import BatchResult, QueryResult

__all__ = [
    "BatchResult",
    "FunctionSpan",
    "GraphData",
    "ParsedFunction",
    "ParsedModule",
    "QueryResult",
]
