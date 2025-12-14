"""Hexagonal architecture port interfaces for the graphs package.

This package defines data transfer objects (DTOs) and deprecated Protocol-based
interfaces. New code should use the resource classes directly.

Data Types (Active)
-------------------
- BatchResult, QueryResult: Storage operation results
- GraphData: Lightweight graph data transfer object
- ParsedFunction, ParsedModule: Parsed source code representations
- FunctionSpan: Unified function span (via re-export from catalog)

Deprecated Protocols
--------------------
- StoragePort: Use StorageResource from resources.storage
- CatalogPort: Use CatalogService from graphs.catalog
- EnginePort: Use GraphResource from resources.graphs

.. deprecated:: 5.0.0
    The Protocol classes (StoragePort, CatalogPort, EnginePort) are deprecated.
    Use the corresponding resource classes directly for dependency injection.

Migration Guide
---------------
Old::

    from codeintel.graphs.ports import StoragePort, CatalogPort


    def process(storage: StoragePort, catalog: CatalogPort) -> None: ...

New::

    from codeintel.graphs.resources import StorageResource
    from codeintel.graphs.catalog import CatalogService


    def process(storage: StorageResource, catalog: CatalogService) -> None: ...
"""

from __future__ import annotations

from codeintel.graphs.catalog import FunctionSpan
from codeintel.graphs.ports.catalog import CatalogPort, FunctionSpanData
from codeintel.graphs.ports.engine import EnginePort, GraphData
from codeintel.graphs.ports.parsing import (
    ParsedFunction,
    ParsedModule,
)
from codeintel.graphs.ports.storage import BatchResult, QueryResult, StoragePort

__all__ = [
    "BatchResult",
    "CatalogPort",
    "EnginePort",
    "FunctionSpan",
    "FunctionSpanData",
    "GraphData",
    "ParsedFunction",
    "ParsedModule",
    "QueryResult",
    "StoragePort",
]
