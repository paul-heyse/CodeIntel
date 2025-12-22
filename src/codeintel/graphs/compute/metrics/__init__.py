"""Pure graph metric computation functions.

This package contains stateless functions for computing graph metrics
without any database or file I/O. These functions decouple metric
computation from the analytics subsystem.

Modules
-------
- bipartite: Bipartite graph metrics (degree centrality, weighted projection)
- cfg: Control flow graph metrics (dominator tree, dominance frontier)
- community: Community detection (greedy modularity, Louvain, label propagation)
- components: SCC, connected components, bridges, articulation points
- dfg: Data flow graph metrics (path lengths, def-use chains)
- paths: Path-related metrics (simple path counting, reachability)
- structural: Structural metrics (clustering, triangles, core number, structural holes)

Centrality Functions
--------------------
Centrality functions (PageRank, betweenness, closeness, harmonic, eigenvector)
are now in ``codeintel.core.compute.centrality``.

Example
-------
```python
import networkx as nx
from codeintel.graphs.compute.metrics import components, structural
from codeintel.core.compute.centrality import compute_pagerank

graph = nx.DiGraph()


pagerank = compute_pagerank(graph)
sccs = components.find_strongly_connected(graph)
clustering = structural.compute_clustering_coefficient(graph)
```
"""

from __future__ import annotations

from codeintel.graphs.compute.metrics import (
    bipartite,
    cfg,
    community,
    components,
    dfg,
    paths,
    statistics,
    structural,
)

__all__ = [
    "bipartite",
    "cfg",
    "community",
    "components",
    "dfg",
    "paths",
    "statistics",
    "structural",
]
