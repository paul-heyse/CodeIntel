"""Concrete adapter implementations for port interfaces.

This package provides implementations of the port protocols defined in
`codeintel.graphs.ports`, connecting abstract interfaces to concrete
infrastructure (DuckDB, LibCST, NetworkX, etc.).

Adapters
--------
- DuckDBStorageAdapter: StoragePort implementation using DuckDB
- LibCSTParsingAdapter: ParsingPort implementation using LibCST
- NxEngineAdapter: EnginePort implementation using NxGraphEngine
- CallgraphPersistence: Persistence operations for call graph edges

Example
-------
```python
from codeintel.graphs.adapters import (
    DuckDBStorageAdapter,
    LibCSTParsingAdapter,
)

storage = DuckDBStorageAdapter(gateway, repo_root)
parser = LibCSTParsingAdapter()

# Use adapters with pure computation functions
source = storage.read_source("module.py")
result = parser.parse_module(source)
```
"""

from __future__ import annotations

from codeintel.graphs.adapters.callgraph_persistence import (
    dedupe_edge_rows,
    default_edge_key,
    persist_call_graph_edges,
)
from codeintel.graphs.adapters.duckdb_storage import DuckDBStorageAdapter
from codeintel.graphs.adapters.libcst_parsing import LibCSTParsingAdapter
from codeintel.graphs.adapters.nx_engine_adapter import NxEngineAdapter

__all__ = [
    "DuckDBStorageAdapter",
    "LibCSTParsingAdapter",
    "NxEngineAdapter",
    "dedupe_edge_rows",
    "default_edge_key",
    "persist_call_graph_edges",
]
