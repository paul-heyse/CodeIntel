"""Data transfer objects (DTOs) for the graphs package.

This package defines lightweight data types for transferring data
between layers without coupling to specific implementations.

Data Types
----------
- BatchResult, QueryResult: Storage operation results
- GraphData: Lightweight graph data transfer object
- ParsedFunction, ParsedModule: Parsed source code representations
- FunctionSpan: Unified function span (from core.catalog)

Usage
-----
For dependency injection, use the resource classes from ``graphs.resources``
or service classes from ``core.catalog`` directly.

Example
-------
```python
from codeintel.build.graphs.resources import StorageResource
from codeintel.core.catalog import CatalogService


def process(storage: StorageResource, catalog: CatalogService) -> None: ...
```
"""

from __future__ import annotations

from codeintel.build.graphs.ports.engine import GraphData
from codeintel.build.graphs.ports.parsing import (
    ParsedFunction,
    ParsedModule,
)
from codeintel.core.catalog import FunctionSpan
from codeintel.core.ports.storage import BatchResult, QueryResult

__all__ = [
    "BatchResult",
    "FunctionSpan",
    "GraphData",
    "ParsedFunction",
    "ParsedModule",
    "QueryResult",
]
