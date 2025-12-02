"""Pure graph metric computation functions.

This package contains stateless functions for computing graph metrics
without any database or file I/O. These functions decouple metric
computation from the analytics subsystem.

Modules
-------
- centrality: PageRank, betweenness, closeness, degree centrality
- components: SCC, connected components, bridges, articulation points
- coupling: Coupling metrics, community detection

Example
-------
```python
import networkx as nx
from codeintel.graphs.compute.metrics import centrality, components

graph = nx.DiGraph()
# ... populate graph ...

pagerank = centrality.compute_pagerank(graph)
sccs = components.find_strongly_connected(graph)
```
"""

from __future__ import annotations

from codeintel.graphs.compute.metrics import centrality, components, coupling

__all__ = [
    "centrality",
    "components",
    "coupling",
]
