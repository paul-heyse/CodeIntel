# Graph Plugin System Decoupling Overview

## Purpose

This document provides a high-level overview of the work required to fully decouple the graph plugin system from the legacy analytics modules, enabling the complete removal of:
- `src/codeintel/analytics/plugins.py`
- `src/codeintel/analytics/plugin_runtime.py`

---

## Current State

### Dependency Chain

```
analytics/plugins.py                    analytics/plugin_runtime.py
        ↓                                        ↓
   AnalyticsPlugin                        AnalyticsPlanRequest
   AnalyticsExecutionContext              AnalyticsRunContext  
   ResourceHints                          plan_analytics_plugin_run
   register_analytics_plugin              run_analytics_plugins
        ↓                                        ↓
    graphs/plugins.py                    graph_service_runtime.py
        ↓                                        ↓
  graphs/runtime/execution.py            GraphServiceRuntime
  graphs/runtime/manifest.py
  graphs/runtime/analytics_adapter.py
```

### Key Integration Point

The main coupling is in `graphs/plugins.py`:

```python
# graphs/plugins.py wraps graph plugins as analytics plugins
def graph_metric_plugin_to_analytics(plugin: GraphMetricPlugin) -> AnalyticsPlugin:
    """Wrap a GraphMetricPlugin as a generic AnalyticsPlugin."""
    ...
```

This allows graph plugins to be executed through the analytics runtime, but creates a tight coupling with legacy types.

---

## High-Level Work Areas

### 1. Standalone Graph Execution Runtime

**Goal**: Graph plugins should execute independently, not as wrapped analytics plugins.

**What's Needed**:
- Extract `graphs/runtime/execution.py` to use only graph-specific types
- Remove the `graph_metric_plugin_to_analytics()` bridge function
- Create a dedicated `GraphPluginExecutor` (similar to new `core/executor.py`)
- Define a `GraphPluginExecutionContext` that doesn't inherit from or wrap analytics types

**Current Files Affected**:
- `graphs/runtime/execution.py`
- `graphs/runtime/planning.py`
- `graphs/runtime/telemetry.py`
- `graphs/plugins.py`

---

### 2. Graph Service Runtime Independence

**Goal**: `GraphServiceRuntime` should use graph-native planning and execution.

**What's Needed**:
- Replace imports from `plugin_runtime.py` with graph-native equivalents
- Create `GraphPlanRequest` and `GraphRunContext` types (graph-specific)
- Implement `plan_graph_plugin_run()` and `run_graph_plugins()` functions
- These should mirror the analytics bridge functions but for graph plugins only

**Current Files Affected**:
- `graph_service_runtime.py`
- Possibly `serving/mcp/architecture_tools.py`

---

### 3. Modernize graphs/plugins.py

**Goal**: Complete the modernization started in `graphs/core/`.

**What's Needed**:
- Move all graph plugin definitions from `graphs/plugins.py` to use the new `@graph_plugin` decorator in `graphs/core/`
- Replace `GraphMetricPlugin` with `GraphPluginProtocol` from `graphs/core/protocol.py`
- Replace `GraphMetricExecutionContext` with a context from `graphs/core/`
- Migrate `GraphRuntimeScratch` to `graphs/core/`
- Remove dependency on `analytics/plugins.py` entirely

**Current Files Affected**:
- `graphs/plugins.py` (to be refactored/replaced)
- `graphs/core/protocol.py` (may need enhancements)
- `graphs/core/registry.py` (may need enhancements)

---

### 4. Update Graph Plugin Consumers

**Goal**: All consumers should import from `graphs/core/` not `graphs/plugins.py`.

**Consumers to Update**:
- `graphs/runtime/execution.py`
- `graphs/runtime/manifest.py`
- `graphs/runtime/analytics_adapter.py`
- `graphs/runtime/planning.py`
- `graphs/runtime/telemetry.py`
- `graphs/runtime/model.py`
- `graphs/catalog.py`
- `tests/analytics/test_graph_plugins.py`
- `tests/analytics/test_graph_plugin_options.py`
- Any external consumers using graph plugins

---

### 5. Remove Analytics Adapter Layer

**Goal**: Eliminate the `graphs/runtime/analytics_adapter.py` translation layer.

**What's Needed**:
- Graph plugins should have their own reporting format
- Remove `_meta_from_graph_record`, `analytics_to_graph_run`, `graph_run_to_analytics`
- Create native graph execution reporting that doesn't need translation

**Current Files Affected**:
- `graphs/runtime/analytics_adapter.py` (to be removed)
- `graph_service_runtime.py` (update to not use adapter)

---

### 6. Update Pipeline Integration

**Goal**: Pipeline should call graph system directly, not through analytics layer.

**What's Needed**:
- Update `pipeline/orchestration/steps_analytics.py` graph-related steps
- Use direct graph plugin execution instead of wrapped analytics execution
- Maintain the same external interface (step configs, reports)

**Current Files Affected**:
- `pipeline/orchestration/steps_analytics.py`
- `pipeline/orchestration/steps_graphs.py` (if exists)

---

## Estimated Scope

| Work Area | Complexity | Files Affected |
|-----------|------------|----------------|
| Standalone Graph Execution | Medium | 4-5 files |
| Graph Service Runtime | Medium | 2-3 files |
| Modernize graphs/plugins.py | High | 3-4 files |
| Update Graph Plugin Consumers | Medium | 6-8 files |
| Remove Analytics Adapter | Low | 2-3 files |
| Update Pipeline Integration | Medium | 1-2 files |

**Total Estimated Files**: 18-25 files

---

## Migration Strategy Options

### Option A: Big Bang Migration
- Modernize all graph infrastructure at once
- Higher risk, but cleaner cutover
- Requires comprehensive test coverage before migration

### Option B: Incremental Migration
1. First, enhance `graphs/core/` to support all graph plugin features
2. Migrate graph plugins one-by-one to new protocol
3. Create parallel graph execution runtime
4. Update consumers to use new imports
5. Remove legacy `graphs/plugins.py` and analytics dependencies
6. Finally delete `analytics/plugins.py` and `plugin_runtime.py`

### Recommended: Option B (Incremental)
- Lower risk, easier to test and validate
- Can be done alongside other work
- Allows rollback at each step

---

## Prerequisites

Before starting this work:
1. ✅ Analytics core architecture complete (done)
2. ✅ Graph plugins modernized protocol defined in `graphs/core/` (done)
3. Comprehensive test coverage for graph plugin execution
4. Inventory of all graph plugin consumers (internal and external)

---

## Success Criteria

The migration is complete when:
1. `analytics/plugins.py` can be deleted with no import errors
2. `analytics/plugin_runtime.py` can be deleted with no import errors
3. All graph plugins execute through `graphs/core/` infrastructure
4. All tests pass
5. No runtime regressions in graph metric computation
6. Pipeline steps work correctly with new graph infrastructure

---

## Open Questions

1. **External consumers**: Are there external packages that import from `graphs/plugins.py`?
2. **Entry points**: Are there setuptools entry points defined for `codeintel.graph_metric_plugins`?
3. **Contract compatibility**: Do graph plugin results need to maintain exact format compatibility?
4. **Telemetry**: Should graph telemetry remain compatible with analytics telemetry?

---

*This is a high-level overview, not a detailed implementation plan. A full plan would be created when this work is prioritized.*

