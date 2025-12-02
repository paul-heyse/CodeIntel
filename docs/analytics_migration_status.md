# Analytics Architecture Migration Status

## 1. Target End State Architecture

### 1.1 Architectural Layers

The new analytics architecture follows a clean layered design that separates concerns:

```
┌──────────────────────────────────────────────────────────────┐
│                    Plugins (Thin Orchestration)              │
│                 codeintel/analytics/core/plugins/            │
│   - Minimal code that wires together resources and compute   │
│   - Uses ctx.require(ProviderType) for all resource access   │
├──────────────────────────────────────────────────────────────┤
│                     Resource Providers                       │
│                  codeintel/analytics/resources/              │
│   - GraphProvider, CatalogProvider, AstProvider              │
│   - AnalyticsContextProvider (legacy bridge, to be removed)  │
│   - Lazy loading via ResourceRegistry                        │
├──────────────────────────────────────────────────────────────┤
│                   Pure Computation Layer                     │
│                   codeintel/analytics/compute/               │
│   - Side-effect-free functions                               │
│   - No I/O, no database access                               │
│   - Easily testable in isolation                             │
├──────────────────────────────────────────────────────────────┤
│                    Persistence Adapters                      │
│                   codeintel/analytics/adapters/              │
│   - Database read/write operations                           │
│   - Table-specific adapters with delete_scope/insert_rows    │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Resource Access Pattern

**Target Pattern:**

```python
def compute(self, ctx: PluginExecutionContext) -> dict[str, int]:
    # Access resources via typed registry
    graph_provider = ctx.require(GraphProvider)
    catalog_provider = ctx.require(CatalogProvider)
    
    # Use pure compute functions
    from codeintel.analytics.compute.functions import compute_complexity
    metrics = compute_complexity(ast_nodes)
    
    # Persist via adapters
    from codeintel.analytics.adapters.functions import FunctionMetricsAdapter
    adapter = FunctionMetricsAdapter()
    return {"analytics.function_metrics": adapter.persist(ctx.gateway, metrics)}
```

### 1.3 Key Principles

1. **No direct context access** - All resources via `ctx.require(ProviderType)`
2. **Pure computation separated** - Business logic in `compute/` with no I/O
3. **Adapters for persistence** - All database operations in `adapters/`
4. **Lazy resource loading** - Resources loaded on first access via providers
5. **Middleware for cross-cutting concerns** - Logging, metrics, tracing via middleware chain

---

## 2. Recently Completed Work

### 2.1 Phase 1-4 (Initial Architecture)

| Component | Status | Location |
|-----------|--------|----------|
| Pure Computation Layer | ✅ Complete | `analytics/compute/` |
| - functions/complexity.py | ✅ | Cyclomatic complexity |
| - functions/typedness.py | ✅ | Type annotation coverage |
| - functions/signatures.py | ✅ | Signature analysis |
| - functions/loc.py | ✅ | Lines of code metrics |
| Persistence Adapters | ✅ Complete | `analytics/adapters/` |
| - base.py | ✅ | BatchAdapter, DeleteScope |
| - functions.py | ✅ | FunctionMetricsAdapter |
| Resource Providers | ✅ Complete | `analytics/resources/` |
| - protocol.py | ✅ | ResourceProvider, LazyResource |
| - registry.py | ✅ | ResourceRegistry |
| - graphs.py | ✅ | GraphProvider |
| - catalog.py | ✅ | CatalogProvider |
| - asts.py | ✅ | AstProvider |
| Plugin Middleware | ✅ Complete | `core/plugins/middleware/` |
| - protocol.py | ✅ | MiddlewareChain |
| - logging.py | ✅ | LoggingMiddleware |
| - metrics.py | ✅ | MetricsMiddleware |
| - tracing.py | ✅ | TracingMiddleware |
| Plugin Groups | ✅ Complete | `core/plugins/groups/` |
| Dataset Pipeline | ✅ Complete | `analytics/pipeline/` |
| - protocol.py | ✅ | DatasetSpec |
| - contracts.py | ✅ | DatasetContract |
| - lineage.py | ✅ | LineageStore |
| - scheduler.py | ✅ | PipelineScheduler |

### 2.2 Migration Infrastructure (Phases A-J)

| Task | Status | Details |
|------|--------|---------|
| Convert functions/typedness.py to shim | ✅ Complete | Re-exports from compute layer |
| Update functions/__init__.py lazy imports | ✅ Complete | Points to compute layer |
| Create AnalyticsContextProvider | ✅ Complete | `resources/analytics_context.py` |
| Add from_resources() to AnalyticsContext | ✅ Complete | Factory from ResourceRegistry |
| Add deprecation to build_analytics_context() | ✅ Complete | Warning added |
| Update plugin base classes with fallbacks | ✅ Complete | CatalogRequiringPlugin, etc. |
| Create compute/dependencies/ | ✅ Complete | detection.py, classification.py |
| Create adapters/dependencies.py | ✅ Complete | DependencyCallAdapter |
| Create compute/profiles/ | ✅ Complete | aggregation.py, features.py |
| Create compute/graphs/ | ✅ Complete | centrality.py, statistics.py |
| Update pipeline_bridge.py with resources | ✅ Complete | Registers providers |
| Add middleware to run_analytics_plugins() | ✅ Complete | LoggingMiddleware, MetricsMiddleware |
| Add resources to RecipeExecutionContext | ✅ Complete | resources: ResourceRegistry field |
| Update execute_plugin for resources | ✅ Complete | Passes registry through |
| Enhance PluginTestHarness | ✅ Complete | with_resources(), with_resource() |
| Add with_graph_provider(), with_catalog_provider() | ✅ Complete | Convenience methods |
| Delete duplicate span_resolver.py | ✅ Complete | Removed |
| Update AGENTS.md | ✅ Complete | New architecture documented |
| Create migration guide | ✅ Complete | docs/analytics_migration_guide.md |

---

## 3. Remaining Work for Full Migration

### 3.1 Phase 1: Plugin Migration (16 files)

All plugins need to be updated from direct context access to `ctx.require()`:

**Function Plugins (5 files):**

- [ ] `core/plugins/functions/metrics.py` - Uses `ctx.analytics_context`
- [ ] `core/plugins/functions/effects.py` - Uses `ctx.analytics_context`, `ctx.graph_runtime`
- [ ] `core/plugins/functions/contracts.py` - Uses `ctx.analytics_context`, `ctx.graph_runtime`
- [ ] `core/plugins/functions/history.py` - Uses `ctx.analytics_context`
- [ ] `core/plugins/functions/ast_features.py` - Uses `ctx.analytics_context`, `ctx.graph_runtime`

**Graph/Coverage Plugins (3 files):**

- [ ] `core/plugins/graphs/core_metrics.py` - Uses `ctx.analytics_context`
- [ ] `core/plugins/coverage/functions.py` - Uses `ctx.analytics_context`
- [ ] `core/plugins/coverage/test_edges.py` - Uses `has_graph_runtime()`

**Domain Plugins (8 files):**

- [ ] `core/plugins/dependencies/external.py`
- [ ] `core/plugins/subsystems/build.py`
- [ ] `core/plugins/profiles/build.py`
- [ ] `core/plugins/data_models/usage.py`
- [ ] `core/plugins/config_data_flow/compute.py`
- [ ] `core/plugins/entrypoints/build.py`
- [ ] `core/plugins/semantic_roles/compute.py`
- [ ] `core/plugins/risk/factors.py`

### 3.2 Phase 2: Domain Module Extraction (10 modules)

Extract pure computation and use adapters:

| Module | Lines | Current State | Action Required |
|--------|-------|---------------|-----------------|
| `dependencies/core.py` | 723 | Mixed I/O | Refactor to use compute/adapters |
| `profiles/__init__.py` | ~400 | Uses AnalyticsContext | Create adapters/profiles.py |
| `subsystems/materialize.py` | ~300 | Direct DB + context | Create compute/subsystems/ |
| `data_model_usage.py` | 582 | Mixed I/O | Create compute/data_models/ |
| `entrypoints/core.py` | ~400 | ensure_analytics_context | Use resource providers |
| `semantic_roles/core.py` | ~300 | Direct context | Create compute/semantic_roles/ |
| `coverage_analytics.py` | 216 | Uses AnalyticsContext | Use providers |
| `ast_features/extract.py` | ~400 | ensure_analytics_context | Use AstProvider |
| `cfg_dfg/materialize.py` | ~300 | Uses AnalyticsContext | Use providers |
| `graphs/config_data_flow.py` | ~400 | ensure_analytics_context | Use providers |

### 3.3 Phase 3: Graph Runtime Consolidation (33 files)

Update imports from `graph_runtime.py` to use `GraphProvider`:

**Analytics Domain (18 files):**

- [ ] `functions/function_effects.py`
- [ ] `functions/function_contracts.py`
- [ ] `dependencies/core.py`
- [ ] `data_model_usage.py`
- [ ] `entrypoints/core.py`
- [ ] `subsystems/materialize.py`
- [ ] `semantic_roles/core.py`
- [ ] `tests/graph_metrics.py`
- [ ] `graphs/config_graph_metrics.py`
- [ ] `graphs/module_graph_metrics_ext.py`
- [ ] `graphs/graph_metrics_ext.py`
- [ ] `graphs/graph_stats.py`
- [ ] `graphs/graph_metrics.py`
- [ ] `graphs/config_data_flow.py`
- [ ] `graphs/symbol_graph_metrics.py`
- [ ] `graphs/subsystem_graph_metrics.py`
- [ ] `core/pipeline_bridge.py`
- [ ] `core/execution_context.py`

**External Modules (15 files):**

- [ ] `cli/main.py`
- [ ] `serving/bootstrap.py`
- [ ] `serving/services/wiring.py`
- [ ] `pipeline/orchestration/core.py`
- [ ] `graphs/validation/findings.py`
- [ ] `graphs/validation/runner.py`
- [ ] `graphs/plugins/validation.py`
- [ ] `graphs/plugins/metrics/core.py`
- [ ] `graphs/plugins/metrics/secondary.py`
- [ ] `graphs/core/adapters.py`
- [ ] `recipes/executor.py`
- [ ] `core/base.py`
- [ ] `resources/analytics_context.py`
- [ ] 2 additional files

### 3.4 Phase 4: Legacy Function Removal (9 files)

Remove `ensure_analytics_context` usage:

- [ ] `functions/function_contracts.py`
- [ ] `dependencies/core.py`
- [ ] `data_model_usage.py`
- [ ] `entrypoints/core.py`
- [ ] `ast_features/extract.py`
- [ ] `semantic_roles/core.py`
- [ ] `core/plugins/functions/ast_features.py`
- [ ] `graphs/config_data_flow.py`
- [ ] `context.py` - Remove the function definition

### 3.5 Phase 5: Context Cleanup

**Remove from PluginExecutionContext:**

```python
# Legacy fields to remove:
_graph_runtime: GraphRuntime | None
_graph_runtime_factory: Callable[[], GraphRuntime] | None
_catalog_provider: FunctionCatalogProvider | None
_catalog_factory: Callable[[], FunctionCatalogProvider] | None
_analytics_context: AnalyticsContext | None
_analytics_context_factory: Callable[[], AnalyticsContext] | None

# Legacy properties to remove:
@property graph_runtime
@property catalog
@property analytics_context

# Legacy methods to remove:
has_graph_runtime()
has_catalog()
has_analytics_context()
```

**Remove from PluginExecutionContextBuilder:**

```python
with_graph_runtime()
with_catalog()
with_analytics_context()
```

### 3.6 Phase 6: Base Class Cleanup

Remove legacy fallbacks from:

- [ ] `CatalogRequiringPlugin.get_catalog()` - Remove legacy fallback
- [ ] `GraphRuntimeRequiringPlugin.get_graph_runtime()` - Remove legacy fallback
- [ ] `AnalyticsContextRequiringPlugin` - Deprecate entire class

### 3.7 Phase 7: Pipeline/Recipe Updates

- [ ] `pipeline_bridge.py` - Remove legacy `with_*` calls
- [ ] `recipes/executor.py` - Remove legacy fields from RecipeExecutionContext
- [ ] `pipeline/orchestration/core.py` - Update to use resource providers

### 3.8 Phase 8: External Module Updates

- [ ] `cli/main.py` - Update GraphRuntime usage
- [ ] `serving/bootstrap.py` - Use resource providers
- [ ] `serving/services/wiring.py` - Update DI configuration
- [ ] `graphs/validation/*.py` (3 files) - Update validation modules
- [ ] `graphs/plugins/metrics/*.py` (2 files) - Update metrics plugins
- [ ] `graphs/core/adapters.py` - Update graph adapters

### 3.9 Phase 9: Test Migration

**Test Helpers:**

- [ ] `tests/_helpers/plugin_harness.py` - Remove legacy fields
- [ ] `tests/_helpers/config_builders.py` - Use resource providers

**Analytics Tests (14 files):**

- [ ] `test_graph_runtime_cache.py`
- [ ] `test_runtime_pool.py`
- [ ] `test_feature_flags_behavior.py`
- [ ] `test_graph_metrics_runtime_reuse.py`
- [ ] `test_graph_metric_filters_integration.py`
- [ ] `test_graph_feature_flags.py`
- [ ] `test_backend_resource_runtime.py`
- [ ] `test_backend_selection.py`
- [ ] `test_validation.py`
- [ ] `test_graph_validation_catalog.py`
- [ ] `test_validation_flags.py`
- [ ] 3 additional test files

### 3.10 Phase 10: Final Cleanup

- [ ] Delete `analytics/graph_service.py` (after updating 2 imports)
- [ ] Update `analytics/__init__.py` exports
- [ ] Mark migration guide as complete
- [ ] Remove deprecated export references

---

## 4. Summary Statistics

| Category | Completed | Remaining |
|----------|-----------|-----------|
| New Infrastructure Files | 25+ | 0 |
| Plugin Migrations | 0 | 16 |
| Domain Module Extractions | 4 | 10 |
| Import Updates | 0 | 33 |
| Legacy Function Removals | 0 | 9 |
| Context Field Removals | 0 | 12 |
| Test Updates | 1 | 14 |
| **Total Files to Modify** | ~30 | ~50+ |

---

## 5. Migration Validation Commands

```bash
# Verify no legacy patterns remain after migration
grep -r "ctx.analytics_context" src/codeintel/analytics/
grep -r "ctx.graph_runtime" src/codeintel/analytics/
grep -r "ensure_analytics_context" src/codeintel/analytics/
grep -r "has_graph_runtime\|has_catalog\|has_analytics_context" src/codeintel/analytics/

# Run quality checks
uv run ruff check --fix
uv run pyright --warnings --pythonversion=3.13
uv run pytest -q
```

---

## 6. File Locations Reference

### New Architecture Components

| Component | Path |
|-----------|------|
| Resource Providers | `src/codeintel/analytics/resources/` |
| Pure Computation | `src/codeintel/analytics/compute/` |
| Persistence Adapters | `src/codeintel/analytics/adapters/` |
| Plugin Middleware | `src/codeintel/analytics/core/plugins/middleware/` |
| Plugin Groups | `src/codeintel/analytics/core/plugins/groups/` |
| Dataset Pipeline | `src/codeintel/analytics/pipeline/` |

### Legacy Components (to be migrated/removed)

| Component | Path | Status |
|-----------|------|--------|
| GraphRuntime | `src/codeintel/analytics/graph_runtime.py` | Keep as internal impl |
| AnalyticsContext | `src/codeintel/analytics/context.py` | Deprecated, use providers |
| graph_service.py | `src/codeintel/analytics/graph_service.py` | To be deleted |
| ensure_analytics_context | `src/codeintel/analytics/context.py` | To be removed |
| build_analytics_context | `src/codeintel/analytics/context.py` | Deprecated |


