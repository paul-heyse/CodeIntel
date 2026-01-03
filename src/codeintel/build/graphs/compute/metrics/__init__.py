"""Pure graph metric computation functions.

This package contains stateless functions for computing graph metrics
without any database or file I/O. These functions decouple metric
computation from the analytics subsystem.

Modules
-------
- bipartite: Bipartite graph metrics (degree centrality, weighted projection)
- centrality: Graph-centric centrality bundles and neighbor statistics
- cfg: Control flow graph metrics (dominator tree, dominance frontier)
- community: Community detection (greedy modularity, Louvain, label propagation)
- components: SCC, connected components, bridges, articulation points
- dfg: Data flow graph metrics (path lengths, def-use chains)
- paths: Path-related metrics (simple path counting, reachability)
- projections: Bipartite projections and metrics
- structural: Structural metrics (clustering, triangles, core number, structural holes)

Centrality Functions
--------------------
Centrality primitives (PageRank, betweenness, closeness, harmonic, eigenvector)
live in ``codeintel.core.compute.centrality`` and are composed into bundles here.

Example
-------
```python
import networkx as nx
from codeintel.build.graphs.compute.metrics import components, structural
from codeintel.core.compute.centrality import compute_pagerank

graph = nx.DiGraph()


pagerank = compute_pagerank(graph)
sccs = components.find_strongly_connected(graph)
clustering = structural.compute_clustering_coefficient(graph)
```
"""

from __future__ import annotations

from codeintel.build.graphs.compute.metrics import (
    bipartite,
    centrality,
    cfg,
    community,
    components,
    conversions,
    dfg,
    paths,
    projections,
    statistics,
    structural,
    types,
)

__all__ = [
    "bipartite",
    "centrality",
    "cfg",
    "community",
    "components",
    "conversions",
    "dfg",
    "paths",
    "projections",
    "statistics",
    "structural",
    "types",
]
