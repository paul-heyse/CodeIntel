"""Graph engine abstractions for analytics consumers.

This package provides the backend-agnostic interface for building and
caching analytics graphs, along with rustworkx-backed implementations.

Key Components
--------------
- protocol: GraphEngine protocol and GraphKind enumeration
- nx_engine: GraphEngine implementation (legacy name; rustworkx-backed loaders)
- cache: Graph caching utilities
- views: SQL-to-rustworkx loaders

Example
-------
```python
from codeintel.build.graphs.engine import GraphEngine, GraphKind, RxGraphEngine

engine = RxGraphEngine(gateway, snapshot)
call_graph = engine.call_graph()
import_graph = engine.import_graph()
```
"""

from __future__ import annotations

from codeintel.build.graphs.engine.backend import BackendEnablement, maybe_enable_nx_gpu
from codeintel.build.graphs.engine.cache import GraphCache
from codeintel.build.graphs.engine.factory import EngineBuildOptions, build_graph_engine
from codeintel.build.graphs.engine.nx_engine import NxGraphEngine
from codeintel.build.graphs.engine.protocol import GraphEngine, GraphKind
from codeintel.build.graphs.engine.rx_engine import RxGraphEngine
from codeintel.build.graphs.engine.views import (
    load_call_graph,
    load_config_module_bipartite,
    load_import_graph,
    load_symbol_function_graph,
    load_symbol_module_graph,
)

__all__ = [
    "BackendEnablement",
    "EngineBuildOptions",
    "GraphCache",
    "GraphEngine",
    "GraphKind",
    "NxGraphEngine",
    "RxGraphEngine",
    "build_graph_engine",
    "load_call_graph",
    "load_config_module_bipartite",
    "load_import_graph",
    "load_symbol_function_graph",
    "load_symbol_module_graph",
    "maybe_enable_nx_gpu",
]
