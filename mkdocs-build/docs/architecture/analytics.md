# Analytics

The analytics module (`codeintel.analytics`) provides plugin-based computation
of code metrics, profiles, and insights.

## Responsibility

- Compute function-level metrics (complexity, size, coupling)
- Build module and subsystem profiles
- Assess risk factors and hotspots
- Analyze test coverage relationships
- Extract semantic roles and entrypoints

## Architecture

```
┌─────────────────────────────────────────┐
│      Pipeline / Orchestration            │
│   (pipeline_bridge.py, executor.py)      │
├─────────────────────────────────────────┤
│           Plugin Layer                   │
│       (core/plugins/*.py)                │
├─────────────────────────────────────────┤
│       Resource Providers                 │
│        (resources/*.py)                  │
├─────────────────────────────────────────┤
│       Pure Compute Layer                 │
│         (compute/*.py)                   │
├─────────────────────────────────────────┤
│       Persistence Adapters               │
│        (adapters/*.py)                   │
└─────────────────────────────────────────┘
```

## Key Entrypoints

- [`codeintel.analytics.core.pipeline_bridge`][codeintel.analytics.core.pipeline_bridge] - Pipeline integration
- [`codeintel.analytics.core.executor`][codeintel.analytics.core.executor] - Plugin execution
- [`codeintel.analytics.core.registry`][codeintel.analytics.core.registry] - Plugin discovery

## Plugin Categories

| Category | Examples |
|----------|----------|
| Function Metrics | Complexity, size, effects, contracts |
| Coverage | Function coverage, test edges, behavioral |
| Profiles | Module profiles, test profiles |
| Risk | Hotspots, risk factors |
| Dependencies | External deps, subsystems |
| Semantic | Roles, entrypoints, data models |

## Dependencies

### Reads From

- Ingestion datasets (functions, modules, AST)
- Graph datasets (call graph, import graph)
- Coverage data

### Writes To

- `analytics.*` tables via adapters
- Computed metrics and profiles

### Called By

- [`codeintel.pipeline`][codeintel.pipeline] orchestration
- [`codeintel.serving`][codeintel.serving] for queries

## Extension Points

### Adding a New Plugin

```python
from dataclasses import dataclass
from codeintel.analytics.core.base import TableWriterPlugin

@dataclass
class MyPlugin(TableWriterPlugin):
    plugin_name = "my.plugin"
    output_tables = ("analytics.my_table",)

    def compute(self, ctx):
        # Pure computation logic
        return {"analytics.my_table": row_count}
```

## See Also

- [Detailed Architecture](../../docs/ANALYTICS_ARCHITECTURE.md#part-i-analytics-module)

