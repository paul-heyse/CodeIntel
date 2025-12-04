"""Pure graph metric computation functions.

This package contains stateless functions for computing graph metrics
without any database or file I/O. These functions decouple metric
computation from the analytics subsystem.

Modules
-------
- centrality: PageRank, betweenness, closeness, harmonic, eigenvector centrality
- cfg: Control flow graph metrics (dominator tree, dominance frontier)
- community: Community detection (greedy modularity, Louvain, label propagation)
- components: SCC, connected components, bridges, articulation points
- coupling: Coupling metrics
- dfg: Data flow graph metrics (path lengths, def-use chains)
- structural: Structural metrics (clustering, triangles, core number, structural holes)

Example
-------
```python
import networkx as nx
from codeintel.graphs.compute.metrics import centrality, components, structural

graph = nx.DiGraph()
# ... populate graph ...

pagerank = centrality.compute_pagerank(graph)
sccs = components.find_strongly_connected(graph)
clustering = structural.compute_clustering_coefficient(graph)
```
"""

from __future__ import annotations

from codeintel.graphs.compute.metrics import (
    centrality,
    cfg,
    community,
    components,
    coupling,
    dfg,
    statistics,
    structural,
)

__all__ = [
    "centrality",
    "cfg",
    "community",
    "components",
    "coupling",
    "dfg",
    "statistics",
    "structural",
]
