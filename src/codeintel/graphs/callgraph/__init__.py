"""Call graph construction domain.

This package consolidates all call graph edge collection, resolution,
and persistence logic into focused modules with clear responsibilities.

Key Components
--------------
- resolution: Callee resolution logic, context structures, and import alias handling
- collectors: AST and CST-based call edge collection strategies
- persistence: Edge deduplication and storage

Example
-------
```python
from codeintel.graphs.callgraph import (
    EdgeResolutionContext,
    ResolutionResult,
    collect_aliases,
    collect_edges_ast,
    collect_edges_cst,
    dedupe_edges,
    persist_call_graph_edges,
    resolve_callee,
)
```
"""

from __future__ import annotations

from codeintel.graphs.callgraph.collectors import (
    collect_edges_ast,
    collect_edges_cst,
    extract_callee_ast,
    extract_callee_cst,
)
from codeintel.graphs.callgraph.persistence import (
    dedupe_edges,
    default_edge_key,
    persist_call_graph_edges,
)
from codeintel.graphs.callgraph.resolution import (
    EdgeResolutionContext,
    ResolutionResult,
    build_evidence,
    collect_aliases,
    collect_import_edges,
    handle_import,
    handle_import_from,
    resolve_callee,
    resolve_via_scip,
)

__all__ = [
    "EdgeResolutionContext",
    "ResolutionResult",
    "build_evidence",
    "collect_aliases",
    "collect_edges_ast",
    "collect_edges_cst",
    "collect_import_edges",
    "dedupe_edges",
    "default_edge_key",
    "extract_callee_ast",
    "extract_callee_cst",
    "handle_import",
    "handle_import_from",
    "persist_call_graph_edges",
    "resolve_callee",
    "resolve_via_scip",
]
