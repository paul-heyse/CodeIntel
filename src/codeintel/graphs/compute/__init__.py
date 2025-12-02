"""Pure stateless computation layer for graph operations.

This package contains pure functions that operate on in-memory data
structures without performing any I/O. All database access, file reading,
and persistence is handled by the ports/adapters layer.

Modules
-------
- goid: GOID hash computation
- callgraph: Call edge collection and resolution
- cfg: Control-flow graph construction
- dfg: Data-flow graph construction
- imports: Import analysis
- symbols: Symbol use analysis
- metrics/: Graph metric computations

Example
-------
```python
from codeintel.graphs.compute import goid, callgraph

# Pure computation - no I/O
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

# Edge collection from parsed module
edges = callgraph.collect_edges(parsed_module, function_spans, context)
```
"""

from __future__ import annotations

# Re-export primary compute modules
from codeintel.graphs.compute import callgraph, cfg, dfg, goid, imports, symbols

__all__ = [
    "callgraph",
    "cfg",
    "dfg",
    "goid",
    "imports",
    "symbols",
]
