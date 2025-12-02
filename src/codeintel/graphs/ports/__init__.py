"""Hexagonal architecture port interfaces for the graphs package.

This package defines Protocol-based interfaces that abstract all I/O operations,
enabling pure computation to be separated from infrastructure concerns.

Port Categories
---------------
- StoragePort: Database query and batch operations
- ParsingPort: CST/AST parsing abstractions
- CatalogPort: Function catalog access
- EnginePort: Graph engine access

Example
-------
```python
from codeintel.graphs.ports import StoragePort, ParsingPort


def process_files(storage: StoragePort, parser: ParsingPort) -> list[Edge]:
    source = storage.read_source("module.py")
    module = parser.parse_module(source)
    # Pure computation on parsed module...
```
"""

from __future__ import annotations

from codeintel.graphs.ports.catalog import CatalogPort, FunctionSpanData
from codeintel.graphs.ports.engine import EnginePort, GraphData
from codeintel.graphs.ports.parsing import (
    ParsedFunction,
    ParsedModule,
    ParsingPort,
)
from codeintel.graphs.ports.storage import BatchResult, QueryResult, StoragePort

__all__ = [
    "BatchResult",
    "CatalogPort",
    "EnginePort",
    "FunctionSpanData",
    "GraphData",
    "ParsedFunction",
    "ParsedModule",
    "ParsingPort",
    "QueryResult",
    "StoragePort",
]
