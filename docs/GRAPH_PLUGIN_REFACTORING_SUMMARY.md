# Graph Plugin Architecture Refactoring Summary

## Executive Summary

This document summarizes the comprehensive refactoring of the graph plugin system to achieve:

1. **Decoupling** from the legacy analytics subsystem
2. **Unified plugin protocol** for builders, metrics, and validation
3. **Recipe-driven orchestration** for flexible pipeline composition
4. **Standalone runtime** that operates independently of analytics infrastructure

---

## Work Completed

### Phase 1: Core Infrastructure (graphs/core/)

Created a unified plugin infrastructure with no analytics dependencies:

| File | Purpose |
|------|---------|
| `protocol.py` | `GraphPluginProtocol` - unified interface for all plugin types |
| `context.py` | `GraphExecutionContext` - execution context with storage, engine, scratch |
| `result.py` | `GraphPluginResult` - standard result type with success/failure/skip |
| `registry.py` | `GraphPluginRegistry` - registry with dependency resolution |

**Key Types Introduced:**

- `GraphPluginProtocol` - Protocol with `metadata` property and `execute()` method
- `GraphPluginMetadata` - Metadata including `kind`, `stage`, `produces_tables`, `requires_graphs`
- `GraphPluginKind` - Literal type: `"builder" | "metric" | "validation"`
- `GraphPluginStage` - Processing stages for ordering execution
- `@graph_plugin` decorator - Convenient decorator for functional plugin definition

### Phase 2: Runtime Infrastructure (graphs/runtime/)

Created standalone execution infrastructure:

| File | Purpose |
|------|---------|
| `executor.py` | Plugin execution with retry, timeout, isolation, and telemetry |
| `planning.py` | Builds execution plans with dependency resolution |
| `manifest.py` | Caching/skip logic with content hashing for incremental execution |
| `telemetry.py` | OpenTelemetry integration for spans and metrics |

**Key Capabilities:**

- Process isolation for fault tolerance
- Timeout and retry handling
- Content-based caching for incremental runs
- OpenTelemetry span and metric emission

### Phase 3: Builder Plugins (graphs/plugins/builders/)

Converted existing graph builders to the new plugin system:

| Plugin | Wraps | Produces |
|--------|-------|----------|
| `goid_builder` | `goid_builder.build_goids()` | `core.goids`, `core.goid_crosswalk` |
| `callgraph_builder` | `callgraph_builder.build_call_graph()` | `graph.call_graph_nodes`, `graph.call_graph_edges` |
| `cfg_dfg_builder` | `cfg_builder.build_cfg_and_dfg()` | `graph.cfg_blocks`, `graph.cfg_edges`, `graph.dfg_edges` |
| `import_graph_builder` | `import_graph.build_import_graph()` | `graph.import_modules`, `graph.import_edges` |

**Dependencies:**

- `callgraph_builder` depends on `goid_builder`
- `cfg_dfg_builder` depends on `goid_builder`
- `import_graph_builder` has no dependencies

### Phase 4: Metric Plugins (graphs/plugins/metrics/)

Migrated graph metric plugins from `analytics/graphs/plugins.py`:

| Plugin | File | Computes |
|--------|------|----------|
| `core_graph_metrics` | `core.py` | Centrality, neighbors, components |
| `graph_metrics_functions_ext` | `core.py` | Extended call graph metrics |
| `graph_metrics_modules_ext` | `core.py` | Extended import graph metrics |
| `cfg_metrics` | `secondary.py` | Control-flow graph metrics |
| `dfg_metrics` | `secondary.py` | Data-flow graph metrics |
| `test_graph_metrics` | `secondary.py` | Test-function bipartite metrics |
| `subsystem_graph_metrics` | `secondary.py` | Subsystem-level metrics |
| `graph_stats` | `secondary.py` | Global graph statistics |

### Phase 5: Validation Plugin (graphs/plugins/)

Created validation as a first-class graph plugin:

| Plugin | Purpose |
|--------|---------|
| `graph_validation` | Validates graph integrity, emits warnings |

### Phase 6: Recipe System (graphs/recipes/)

Created a declarative recipe DSL for pipeline composition:

| File | Purpose |
|------|---------|
| `dsl.py` | `GraphRecipe`, `GraphStage` definitions |
| `executor.py` | `RecipeExecutor` for orchestrating stages |
| `builtins.py` | Standard recipes |

**Builtin Recipes:**

- `FULL_GRAPH_RECIPE` - All builders + metrics + validation
- `BUILDERS_ONLY_RECIPE` - Just graph construction
- `METRICS_ONLY_RECIPE` - Metrics on existing graphs
- `INCREMENTAL_RECIPE` - Skip unchanged plugins
- `CALLGRAPH_ONLY_RECIPE` - Minimal call graph
- `IMPORT_GRAPH_ONLY_RECIPE` - Minimal import graph
- `VALIDATION_ONLY_RECIPE` - Just validation

### Phase 7: Cleanup

- Removed `_register_graph_plugins_as_analytics()` from `analytics/graphs/plugins.py`
- Emptied `analytics/graphs/runtime/analytics_adapter.py`
- Updated `docs/GRAPH_DECOUPLING_OVERVIEW.md` with implementation status

---

## End-State Architecture

### Directory Structure

```
src/codeintel/graphs/
├── core/                         # Unified plugin protocol and context
│   ├── __init__.py              # Package exports
│   ├── protocol.py              # GraphPluginProtocol, @graph_plugin decorator
│   ├── context.py               # GraphExecutionContext, GraphRuntimeScratch
│   ├── registry.py              # GraphPluginRegistry with dependency resolution
│   └── result.py                # GraphPluginResult, GraphPluginRunRecord
│
├── runtime/                      # Standalone execution infrastructure
│   ├── __init__.py              # Package exports
│   ├── executor.py              # run_graph_plugin_batch(), BatchContext
│   ├── planning.py              # plan_graph_plugin_run(), PluginExecutionPlan
│   ├── manifest.py              # Caching/skip logic, content hashing
│   └── telemetry.py             # OpenTelemetry integration
│
├── recipes/                      # Declarative recipe DSL
│   ├── __init__.py              # Package exports
│   ├── dsl.py                   # GraphRecipe, GraphStage, graph_recipe()
│   ├── executor.py              # RecipeExecutor
│   └── builtins.py              # Standard recipes
│
├── plugins/                      # All graph plugins
│   ├── __init__.py              # Package exports
│   ├── validation.py            # GraphValidationPlugin
│   ├── builders/                # Graph construction plugins
│   │   ├── __init__.py
│   │   ├── goid.py              # GoidBuilderPlugin
│   │   ├── callgraph.py         # CallGraphBuilderPlugin
│   │   ├── cfg_dfg.py           # CFGDFGBuilderPlugin
│   │   └── import_graph.py      # ImportGraphBuilderPlugin
│   └── metrics/                 # Graph metric plugins
│       ├── __init__.py
│       ├── core.py              # Core metrics
│       └── secondary.py         # Secondary metrics
│
├── engine.py                     # GraphEngine protocol (unchanged)
├── engine_factory.py             # Factory (unchanged)
├── nx_views.py                   # NetworkX views (unchanged)
├── validation.py                 # Validation logic (wrapped by plugin)
└── [existing builders]           # Original builders (wrapped by plugins)
```

### Plugin Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│                    Recipe DSL                            │
│     GraphRecipe → [GraphStage → [plugin_names]]         │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                GraphPluginRegistry                       │
│     resolve dependencies → topological sort              │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              PluginExecutionPlan                         │
│     ordered plugins + settings + options                 │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│             run_graph_plugin_batch()                     │
│     execute each plugin → collect records                │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                RecipeExecutionResult                     │
│     stage_results + overall success + duration           │
└─────────────────────────────────────────────────────────┘
```

### Usage Examples

**Running a Full Pipeline via Recipe:**

```python
from codeintel.graphs.recipes import FULL_GRAPH_RECIPE, RecipeExecutor
from codeintel.graphs.core import GraphExecutionContext

ctx = GraphExecutionContext(
    gateway=gateway,
    snapshot=snapshot,
    engine=engine,
    catalog_provider=catalog_provider,
    scratch=scratch,
    options=None,
    scope=scope,
)

executor = RecipeExecutor(registry=get_graph_registry())
result = executor.execute(FULL_GRAPH_RECIPE, ctx)
print(f"Success: {result.success}, Duration: {result.total_duration_ms}ms")
```

**Creating a Custom Plugin:**

```python
from codeintel.graphs.core import (
    graph_plugin,
    GraphExecutionContext,
    GraphPluginResult,
)

@graph_plugin(
    name="my_custom_builder",
    description="Build a custom graph structure.",
    kind="builder",
    stage="edges",
    depends_on=("goid_builder",),
    produces_tables=("graph.my_edges",),
)
def my_custom_builder(ctx: GraphExecutionContext) -> GraphPluginResult:
    # Build graph and persist to database
    return GraphPluginResult.ok(row_counts={"graph.my_edges": 100})
```

---

## Remaining Work

### ✅ Completed (December 2025)

#### 1. Update Pipeline Integration ✅

The pipeline orchestration steps now use the new infrastructure:

- `pipeline/orchestration/steps_analytics.py` - `GraphMetricsStep` uses `RecipeExecutor`
- `pipeline/orchestration/steps_graphs.py` - Uses new graph registry

#### 2. Update MCP/Serving Tools ✅

Serving tools now import from new locations:

- `serving/mcp/architecture_tools.py` - Uses `plan_graph_plugins` from `graphs.core.registry`

#### 3. Complete Metric Plugin Migration ✅

All metric plugins now exist in the new system:

- `symbol_graph_metrics_modules` - Added to `secondary.py`
- `symbol_graph_metrics_functions` - Added to `secondary.py`
- `config_graph_metrics` - Added to `secondary.py`
- `subsystem_agreement` - Added to `secondary.py`

#### 4. Test Helper Migration ✅

Created new test infrastructure for the new plugin system:

- `tests/_helpers/graph_plugin_packs.py` - New test plugin packs using `graphs.core`
- `tests/analytics/conftest.py` - Added `NewPluginTestHarness` using `RecipeExecutor`

#### 5. Validation Module ✅

`validation.py` uses `GraphRuntime` from `analytics.graph_runtime` which is a utility class for graph access, not part of the deprecated plugin system. This remains intentional.

#### 6. Direct Metric Computation ✅

Metric plugins now call computation functions directly instead of using `importlib.import_module`.

#### 7. Entry Point Registration ✅

Added `pyproject.toml` entry points for `codeintel.graph_plugins` discovery.

#### 8. Parallel Execution ✅

`RecipeExecutor` now implements parallel plugin execution using `ThreadPoolExecutor` for stages marked `parallel=True`.

#### 9. CLI Updates ✅

CLI (`cli/main.py`) now uses:
- `plan_graph_plugins` from `graphs.core.registry`
- `list_graph_plugins` from `graphs.core.registry`
- `DEFAULT_METRIC_PLUGINS` from `graphs.core.protocol`

### Pending - Future Cleanup Phase

#### Legacy Module Deletion

The following modules have deprecation warnings but are retained for backward compatibility during the transition:

- `analytics/graphs/plugins.py` - Has `DeprecationWarning`, still used by tests
- `analytics/graphs/runtime/analytics_adapter.py` - Has `DeprecationWarning`
- `analytics/graph_service_runtime.py` - Still used by legacy tests

These modules can be deleted once all test files are migrated to use the new test infrastructure.

#### Legacy Test Migration

The following test files still test the legacy `GraphServiceRuntime` infrastructure and can be migrated or deleted in a future phase:

- `tests/analytics/test_graph_plugin_policy_runtime.py`
- `tests/analytics/test_graph_service_runtime.py`
- `tests/analytics/test_graph_runtime_execution.py`

### Documentation

- ✅ Updated `GRAPH_PLUGIN_REFACTORING_SUMMARY.md` (this document)
- ⬜ Create user-facing documentation for graph plugin API
- ⬜ Add migration guide for existing consumers

---

## Current Plugin Registration

The registry contains 17 plugins:

**Builders (4):**
- `goid_builder`
- `callgraph_builder`
- `cfg_dfg_builder`
- `import_graph_builder`

**Metrics (12):**
- `core_graph_metrics`
- `graph_metrics_functions_ext`
- `graph_metrics_modules_ext`
- `cfg_metrics`
- `dfg_metrics`
- `test_graph_metrics`
- `subsystem_graph_metrics`
- `graph_stats`
- `symbol_graph_metrics_modules` *(added December 2025)*
- `symbol_graph_metrics_functions` *(added December 2025)*
- `config_graph_metrics` *(added December 2025)*
- `subsystem_agreement` *(added December 2025)*

**Validation (1):**
- `graph_validation`

---

## Success Criteria Status

| Criterion | Status |
|-----------|--------|
| `graphs/core/` provides unified plugin protocol | ✅ Complete |
| `graphs/runtime/` provides standalone execution | ✅ Complete |
| `graphs/plugins/builders/` contains all builder plugins | ✅ Complete |
| `graphs/plugins/metrics/` contains metric plugins | ✅ Complete |
| `graphs/recipes/` provides declarative pipeline composition | ✅ Complete |
| Pipeline steps use graph recipes for orchestration | ✅ Complete |
| MCP/serving tools import from new locations | ✅ Complete |
| CLI imports from new locations | ✅ Complete |
| New test infrastructure created | ✅ Complete |
| Direct metric computation (no importlib) | ✅ Complete |
| Parallel stage execution in RecipeExecutor | ✅ Complete |
| Entry points for plugin discovery | ✅ Complete |
| Legacy modules have deprecation warnings | ✅ Complete |
| All tests pass with new infrastructure | ⚠️ Legacy tests still use old system |
| `analytics/plugins.py` has zero graph-related code | ⬜ Future cleanup |
| `analytics/plugin_runtime.py` has zero graph imports | ⬜ Future cleanup |

---

## Files Created

| Path | Lines | Purpose |
|------|-------|---------|
| `graphs/core/__init__.py` | ~70 | Package exports |
| `graphs/core/protocol.py` | ~280 | Plugin protocol and decorator |
| `graphs/core/context.py` | ~150 | Execution context |
| `graphs/core/registry.py` | ~350 | Plugin registry |
| `graphs/core/result.py` | ~140 | Result types |
| `graphs/runtime/__init__.py` | ~60 | Package exports |
| `graphs/runtime/executor.py` | ~590 | Plugin executor |
| `graphs/runtime/planning.py` | ~420 | Execution planning |
| `graphs/runtime/manifest.py` | ~370 | Caching/manifest |
| `graphs/runtime/telemetry.py` | ~360 | Telemetry |
| `graphs/recipes/__init__.py` | ~50 | Package exports |
| `graphs/recipes/dsl.py` | ~100 | Recipe DSL |
| `graphs/recipes/executor.py` | ~480 | Recipe executor |
| `graphs/recipes/builtins.py` | ~140 | Builtin recipes |
| `graphs/plugins/__init__.py` | ~30 | Package exports |
| `graphs/plugins/validation.py` | ~200 | Validation plugin |
| `graphs/plugins/builders/__init__.py` | ~30 | Package exports |
| `graphs/plugins/builders/goid.py` | ~160 | GOID builder |
| `graphs/plugins/builders/callgraph.py` | ~190 | Call graph builder |
| `graphs/plugins/builders/cfg_dfg.py` | ~190 | CFG/DFG builder |
| `graphs/plugins/builders/import_graph.py` | ~160 | Import graph builder |
| `graphs/plugins/metrics/__init__.py` | ~40 | Package exports |
| `graphs/plugins/metrics/core.py` | ~380 | Core metrics |
| `graphs/plugins/metrics/secondary.py` | ~930 | Secondary metrics (expanded) |
| `tests/_helpers/graph_plugin_packs.py` | ~220 | Test plugin packs for new infrastructure |

---

## Key Design Decisions

### 1. Unified Plugin Protocol

A single `GraphPluginProtocol` supports builders, metrics, and validation with a `kind` discriminator rather than separate hierarchies.

### 2. Graph-Native Context

`GraphExecutionContext` replaces `AnalyticsExecutionContext` dependency, containing only graph-relevant resources.

### 3. Recipe-Driven Orchestration

Declarative recipes define plugin execution order, enabling reusable pipelines without code changes.

### 4. Incremental Execution

Content hashing and manifest tracking enable skipping unchanged plugins for faster incremental runs.

### 5. Fault Isolation

Process isolation and timeout handling prevent individual plugin failures from crashing the entire pipeline.

---

*Generated: December 2, 2025*
*Last updated: December 2, 2025 - Phase A through H implementation complete*
