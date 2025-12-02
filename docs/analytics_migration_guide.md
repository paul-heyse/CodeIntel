# Analytics Architecture Migration Guide

This guide helps plugin authors migrate from the legacy monolithic architecture to the new layered architecture.

## Overview

The new analytics architecture separates concerns into distinct layers:

1. **Compute Layer** (`analytics/compute/`): Pure functions with no I/O
2. **Adapters** (`analytics/adapters/`): Database persistence operations  
3. **Resources** (`analytics/resources/`): Lazy resource loading
4. **Plugins** (`analytics/core/plugins/`): Thin orchestration

## Key Changes

### 1. Resource Access via Registry

**Before (Legacy)**:
```python
def compute(self, ctx: PluginExecutionContext) -> dict[str, int]:
    # Direct access to monolithic context
    graph_runtime = ctx.graph_runtime
    catalog = ctx.catalog
    analytics_context = ctx.analytics_context
```

**After (New)**:
```python
from codeintel.analytics.resources.graphs import GraphProvider
from codeintel.analytics.resources.catalog import CatalogProvider

def compute(self, ctx: PluginExecutionContext) -> dict[str, int]:
    # Access via resource registry
    graph_provider = ctx.require(GraphProvider)
    call_graph = graph_provider.call_graph
    
    catalog_provider = ctx.require(CatalogProvider)
    catalog = catalog_provider.get()
```

### 2. Use Pure Compute Functions

**Before**:
```python
def compute(self, ctx: PluginExecutionContext) -> dict[str, int]:
    # Business logic mixed with I/O
    for func in ctx.analytics_context.function_ast_map.values():
        complexity = self._compute_complexity(func)
        self._write_to_db(ctx.gateway, complexity)
```

**After**:
```python
from codeintel.analytics.compute.functions.complexity import compute_complexity
from codeintel.analytics.adapters.functions import FunctionMetricsAdapter

def compute(self, ctx: PluginExecutionContext) -> dict[str, int]:
    # Pure computation
    ast_provider = ctx.require(AstProvider)
    metrics = [compute_complexity(func) for func in ast_provider.function_asts.values()]
    
    # Persistence via adapter
    adapter = FunctionMetricsAdapter()
    return {"analytics.function_metrics": adapter.persist_batch(ctx.gateway, metrics)}
```

### 3. Middleware for Cross-Cutting Concerns

Logging, metrics, and tracing are now handled by middleware:

```python
from codeintel.analytics.core.executor import PluginExecutor, ExecutionPolicy
from codeintel.analytics.core.plugins.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
)

executor = PluginExecutor(
    policy=ExecutionPolicy(),
    middleware=[LoggingMiddleware(), MetricsMiddleware()],
)
```

## Migration Steps

### Step 1: Update Resource Access

Replace direct context access with `ctx.require()`:

| Old | New |
|-----|-----|
| `ctx.graph_runtime` | `ctx.require(GraphProvider)` |
| `ctx.catalog` | `ctx.require(CatalogProvider).get()` |
| `ctx.analytics_context` | Use specific providers instead |

### Step 2: Extract Pure Logic

Move business logic to `analytics/compute/`:

```
analytics/compute/
├── functions/
│   ├── complexity.py    # Pure complexity computation
│   ├── typedness.py     # Pure typedness computation
│   └── signatures.py    # Pure signature analysis
├── dependencies/
│   ├── detection.py     # Pure dependency detection
│   └── classification.py
└── graphs/
    ├── centrality.py    # Pure graph metrics
    └── statistics.py
```

### Step 3: Use Adapters for Persistence

Create or use existing adapters in `analytics/adapters/`:

```python
from codeintel.analytics.adapters.base import BatchAdapter, DeleteScope

class MyMetricsAdapter(BatchAdapter[MyMetricRow]):
    table_name = "analytics.my_metrics"
    
    def delete_scope(self, gateway, scope: DeleteScope) -> int:
        # Delete rows within scope
        ...
    
    def insert_rows(self, gateway, rows) -> int:
        # Insert computed rows
        ...
```

### Step 4: Update Tests

Use the enhanced test harness with resource support:

```python
from tests._helpers.plugin_harness import PluginTestHarness
from codeintel.analytics.resources.graphs import GraphProvider

def test_my_plugin(analytics_gateway, snapshot):
    graph_provider = GraphProvider(analytics_gateway, snapshot)
    
    result = (
        PluginTestHarness.for_plugin(MyPlugin())
        .with_gateway(analytics_gateway)
        .with_graph_provider(graph_provider)
        .execute()
    )
    
    assert result.success
```

## Deprecation Timeline

| Component | Status | Replacement |
|-----------|--------|-------------|
| `build_analytics_context()` | Deprecated | `AnalyticsContextProvider` |
| `ctx.analytics_context` | Deprecated | Specific resource providers |
| `functions/typedness.py` | Shim | `compute/functions/typedness.py` |

## FAQ

### Q: Can I still use the legacy API?

Yes, the legacy API continues to work during the transition. However, you'll see deprecation warnings. Plan to migrate before the next major release.

### Q: What if my plugin needs AnalyticsContext?

Use `AnalyticsContextProvider` from the registry:

```python
from codeintel.analytics.resources.analytics_context import AnalyticsContextProvider

provider = ctx.require(AnalyticsContextProvider)
context = provider.get()
```

### Q: How do I test with mock resources?

Use the harness's `with_resource()` method:

```python
harness.with_resource(GraphProvider, mock_graph_provider)
```

### Q: Where should new compute functions go?

Add them to the appropriate module under `analytics/compute/`:
- Function analysis: `compute/functions/`
- Dependency analysis: `compute/dependencies/`
- Graph metrics: `compute/graphs/`
- Profile aggregation: `compute/profiles/`

Create new subdirectories as needed for new domains.

