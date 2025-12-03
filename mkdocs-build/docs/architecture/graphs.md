# Graphs

The graphs module (`codeintel.graphs`) builds and analyzes code relationship
graphs including call graphs, import graphs, and control/data flow graphs.

## Responsibility

- Build call graphs from function references
- Build import graphs from module dependencies
- Construct control flow graphs (CFG)
- Construct data flow graphs (DFG)
- Compute graph metrics (centrality, components, PageRank)

## Architecture

```
┌─────────────────────────────────────────┐
│          Runtime Layer                   │
│        (runtime/*.py)                    │
├─────────────────────────────────────────┤
│          Plugin Layer                    │
│   (plugins/builders/, plugins/metrics/)  │
├─────────────────────────────────────────┤
│         Resource Layer                   │
│        (resources/*.py)                  │
├─────────────────────────────────────────┤
│          Engine Layer                    │
│         (engine/*.py)                    │
├─────────────────────────────────────────┤
│         Port-Adapter Layer               │
│    (ports/*.py, adapters/*.py)           │
└─────────────────────────────────────────┘
```

## Key Entrypoints

- [`codeintel.graphs.runtime.executor`][codeintel.graphs.runtime.executor] - Plugin execution
- [`codeintel.graphs.runtime.planning`][codeintel.graphs.runtime.planning] - Execution planning
- [`codeintel.graphs.core.registry`][codeintel.graphs.core.registry] - Plugin registry

## Plugin Types

### Builder Plugins

| Plugin | Output |
|--------|--------|
| `GoidBuilderPlugin` | Global object IDs |
| `CallGraphBuilderPlugin` | Call relationships |
| `ImportGraphBuilderPlugin` | Module imports |
| `CFGDFGBuilderPlugin` | Control/data flow |

### Metric Plugins

| Plugin | Metrics |
|--------|---------|
| Core Metrics | Centrality, components |
| Secondary Metrics | CFG, DFG, test metrics |

## Dependencies

### Reads From

- Ingestion datasets (functions, modules, AST)
- SCIP index for semantic references

### Writes To

- `graph.*` tables
- NetworkX graph structures

### Called By

- [`codeintel.pipeline`][codeintel.pipeline] orchestration
- [`codeintel.analytics`][codeintel.analytics] for graph-based metrics

## Extension Points

### Adding a Graph Builder

```python
from codeintel.graphs.core.protocol import GraphPluginProtocol

class MyGraphBuilder(GraphPluginProtocol):
    plugin_name = "my.builder"
    plugin_kind = "builder"

    def execute(self, ctx):
        # Build graph
        return GraphPluginResult(...)
```

## See Also

- [Detailed Architecture](../../docs/ANALYTICS_ARCHITECTURE.md#part-iii-graphs-module)
- [Graph Decoupling Overview](../../docs/GRAPH_DECOUPLING_OVERVIEW.md)

