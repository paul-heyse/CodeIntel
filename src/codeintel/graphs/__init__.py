"""Graph builders, plugins, and infrastructure for code graph construction and analysis.

This package provides graph construction and analysis capabilities integrated
with the Hamilton build system.

Package Structure
-----------------
Plugins (graphs/plugins/):
- builders: Graph construction plugins (goid, callgraph, cfg_dfg, import_graph)
- metrics: Graph metric computation plugins
- validation: Graph validation plugin

Hexagonal Architecture (Ports, Compute, Resources):
- ports/: Protocol interfaces abstracting I/O (StoragePort, CatalogPort, etc.)
- compute/: Pure stateless computation functions (no I/O)
- resources/: DI container and resource providers

Consolidated Domain Packages:
- catalog/: Function catalog (spans, metadata, service) - unified module
- validation/: Graph validation checks, findings, and orchestration
- engine/: Graph engine protocol, NetworkX implementation, and views

Callgraph logic is in compute/callgraph/ (pure functions and persistence utilities).

Example
-------
```python
from codeintel.build.registry import get_target_graph
from codeintel.graphs.compute.metrics import centrality


graph = get_target_graph()
graph_targets = [t for t in graph.all_targets if t.module == "graphs"]


pagerank = centrality.compute_pagerank(call_graph)
```

Architecture Notes
------------------
The graphs package uses hexagonal architecture:

- Plugins are the orchestration layer, composing resources and compute functions
- Resources provide injectable access to I/O (storage, engine, catalog)
- Compute functions are pure and stateless, taking data and returning data
- Ports define protocol interfaces for abstraction

All builder modules have been consolidated into their corresponding plugins
under plugins/builders/. Pure computation logic is in compute/.

Plugin registration is handled by the build registry in codeintel.build.plugin_registry.
"""

from __future__ import annotations

from codeintel.core.resources import ResourceRegistry
from codeintel.graphs import compute, ports, resources
from codeintel.graphs.catalog import CatalogService, FunctionSpan
from codeintel.graphs.engine import GraphEngine, GraphKind, NxGraphEngine
from codeintel.graphs.resources import ResourceProvider

__all__ = [
    "CatalogService",
    "FunctionSpan",
    "GraphEngine",
    "GraphKind",
    "NxGraphEngine",
    "ResourceProvider",
    "ResourceRegistry",
    "compute",
    "ports",
    "resources",
]
