"""Pure stateless computation layer for graph operations.

This package contains pure functions that operate on in-memory data
structures without performing any I/O. All database access, file reading,
and persistence is handled by the resources layer.

Subpackages
-----------
callgraph/
    Call edge collection, resolution, and deduplication utilities.
    - collection: Edge collection from AST/CST
    - resolution: Callee resolution logic
    - persistence: Deduplication utilities
    - types: CallEdge, ResolutionResult, contexts
metrics/
    Graph metric computations (centrality, community, structural, etc.)
    - centrality: PageRank, betweenness, closeness
    - community: Community detection algorithms
    - components: SCC, connected components
    - structural: Clustering, triangles
    - coupling: Module/function coupling metrics

Modules
-------
cfg
    Control-flow graph construction
dfg
    Data-flow graph construction
goid
    GOID hash computation and URN building
imports
    Import relationship analysis
symbols
    Symbol use analysis

Example
-------
```python
from codeintel.graphs.compute import goid, callgraph

descriptor = goid.GoidDescriptor(
    repo="myrepo",
    commit="abc123",
    language="python",
    rel_path="module.py",
    kind="function",
    qualname="module.func",
    start_line=10,
    end_line=20,
)
goid_hash = goid.compute_goid(descriptor)
urn = goid.build_urn(descriptor)

edges = callgraph.collect_edges(parsed_module, function_spans, context)
```
"""

from __future__ import annotations

from codeintel.graphs.compute import callgraph, cfg, dfg, goid, imports, symbols

__all__ = [
    "callgraph",
    "cfg",
    "dfg",
    "goid",
    "imports",
    "symbols",
]
