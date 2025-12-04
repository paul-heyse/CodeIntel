"""Graph engine abstractions for analytics consumers.

This package provides the backend-agnostic interface for building and
caching analytics graphs, along with the NetworkX-backed implementation.

Key Components
--------------
- protocol: GraphEngine protocol and GraphKind enumeration
- nx_engine: NetworkX-backed GraphEngine implementation
- cache: Graph caching utilities
- views: SQL-to-NetworkX loaders

Example
-------
```python
from codeintel.graphs.engine import (
    GraphEngine,
    GraphKind,
    NxGraphEngine,
)

engine = NxGraphEngine(gateway, snapshot)
call_graph = engine.call_graph()
import_graph = engine.import_graph()
```
"""

from __future__ import annotations

from codeintel.graphs.engine.backend import BackendEnablement, maybe_enable_nx_gpu
from codeintel.graphs.engine.cache import GraphCache
from codeintel.graphs.engine.factory import EngineBuildOptions, build_graph_engine
from codeintel.graphs.engine.nx_engine import NxGraphEngine
from codeintel.graphs.engine.protocol import GraphEngine, GraphKind
from codeintel.graphs.engine.views import (
    load_call_graph,
    load_config_module_bipartite,
    load_import_graph,
    load_symbol_function_graph,
    load_symbol_module_graph,
    load_test_function_bipartite,
)

__all__ = [
    "BackendEnablement",
    "EngineBuildOptions",
    "GraphCache",
    "GraphEngine",
    "GraphKind",
    "NxGraphEngine",
    "build_graph_engine",
    "load_call_graph",
    "load_config_module_bipartite",
    "load_import_graph",
    "load_symbol_function_graph",
    "load_symbol_module_graph",
    "load_test_function_bipartite",
    "maybe_enable_nx_gpu",
]
