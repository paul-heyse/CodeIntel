# PLC0415 Import Resolution Progress

## Status: ✅ COMPLETE

All 47 `PLC0415` errors have been resolved without using any suppressions.

## Context

This document tracks the progress of resolving all `PLC0415` Ruff errors ("`import` should be at the top-level of a file") in the `@analytics` module without using suppressions. The goal was to eliminate lazy/deferred imports through architectural improvements, specifically:

1. **TYPE_CHECKING migration**: Move type-only imports to `if TYPE_CHECKING:` blocks
2. **String-based provider lookups**: Add `require_by_name()` and `has_resource_by_name()` methods to `ResourceRegistry` and `PluginExecutionContext` to enable provider access without runtime imports
3. **Moving imports to top-level**: For imports that don't cause circular dependencies

The original scan found **47 lazy imports across 27 files** in `src/codeintel/analytics/`.

---

## Completed Work

### Phase 1: Standard Library Imports

Moved stdlib imports to top-level in 2 files:

| File | Import | Line |
|------|--------|------|
| `core/plugins/middleware/tracing.py` | `import secrets` | 138 |
| `pipeline/contracts.py` | `import time` | 178 |

### Phase 2: ResourceRegistry Enhancement

Enhanced `src/codeintel/analytics/resources/registry.py` with string-based provider lookup:

**New methods added:**
- `get_by_name(name: str) -> ResourceProvider[Any]` - Get provider by string name
- `has_by_name(name: str) -> bool` - Check if provider exists by name
- `require_by_name(name: str) -> Any` - Load resource by name

**Modified methods:**
- `register()` - Now also stores providers by class name in `_providers_by_name`
- `register_or_replace()` - Same enhancement
- `clear()` - Clears both dictionaries

### Phase 3: PluginExecutionContext Enhancement

Added string-based lookup methods to `src/codeintel/analytics/core/execution_context.py`:

- `require_by_name(name: str) -> Any`
- `has_resource_by_name(name: str) -> bool`

These delegate to the underlying `ResourceRegistry`.

### Phase 4: core/base.py Migration

Migrated 7 lazy imports in `src/codeintel/analytics/core/base.py`:

**Before:**
```python
def get_catalog(self, ctx):
    from codeintel.analytics.resources.catalog import CatalogProvider
    provider = ctx.require(CatalogProvider)
    return provider.get()
```

**After:**
```python
# At top of file, in TYPE_CHECKING block:
if TYPE_CHECKING:
    from codeintel.analytics.resources.catalog import CatalogProvider
    from codeintel.analytics.resources.analytics_context import AnalyticsContextProvider
    from codeintel.analytics.resources.graphs import GraphProvider

# In method:
def get_catalog(self, ctx):
    provider: CatalogProvider = ctx.require_by_name("CatalogProvider")
    return provider.get()
```

**Classes updated:**
- `CatalogRequiringPlugin._validate_resource_requirements()`
- `CatalogRequiringPlugin.get_catalog()`
- `AnalyticsContextRequiringPlugin._validate_resource_requirements()`
- `AnalyticsContextRequiringPlugin.get_analytics_context()`
- `AnalyticsContextRequiringPlugin.get_analytics_context_or_none()`
- `GraphRuntimeRequiringPlugin._validate_resource_requirements()`
- `GraphRuntimeRequiringPlugin.get_graph_runtime()`

### Phase 5: context.py Migration

Migrated 5 lazy imports in `src/codeintel/analytics/context.py`:

- Moved `AstProvider`, `CatalogProvider`, `FeaturesProvider`, `GraphProvider` to `TYPE_CHECKING`
- Updated `AnalyticsContext.from_resources()` to use `registry.require_by_name()` and `registry.has_by_name()`
- Removed duplicate `import warnings` (line 697)

### Phase 6: Plugin Compute Methods Migration

Migrated **17 lazy imports across 15 plugin files** to TYPE_CHECKING pattern:

| Plugin File | Provider(s) |
|-------------|-------------|
| `config_data_flow/compute.py` | `AnalyticsContextProvider` |
| `coverage/functions.py` | `AnalyticsContextProvider` |
| `coverage/test_edges.py` | `CatalogProvider` |
| `data_models/usage.py` | `AnalyticsContextProvider` |
| `dependencies/external.py` | `AnalyticsContextProvider` |
| `entrypoints/build.py` | `AnalyticsContextProvider` |
| `functions/ast_features.py` | `AnalyticsContextProvider` |
| `functions/contracts.py` | `AnalyticsContextProvider` |
| `functions/effects.py` | `AnalyticsContextProvider`, `GraphProvider` |
| `functions/history.py` | `AnalyticsContextProvider` |
| `functions/metrics.py` | `AnalyticsContextProvider` |
| `profiles/build.py` | `AnalyticsContextProvider` |
| `risk/factors.py` | `CatalogProvider` |
| `semantic_roles/compute.py` | `AnalyticsContextProvider` |
| `subsystems/build.py` | `AnalyticsContextProvider`, `GraphProvider` |

### Phase 7: Compute Graph Modules (3 errors) - NetworkX imports ✅

| File | Line | Import |
|------|------|--------|
| `compute/graphs/centrality.py` | 75, 119 | `import networkx as nx` |
| `compute/graphs/statistics.py` | 81 | `import networkx as nx` |

**Fix applied:** Moved `import networkx as nx` to top-level in both files. Since `nx_types.py` in the same subpackage already imports networkx at top-level, there was no benefit to lazy loading.

### Phase 8: Resource Provider Imports (7 errors) ✅

After analysis, confirmed **no circular dependencies** between resource providers and the services they import. Moved all imports to top-level:

| File | Import | Resolution |
|------|--------|------------|
| `resources/analytics_context.py` | `build_analytics_context` | Moved to top-level |
| `resources/asts.py` | `FunctionAstLoadRequest`, `load_function_asts` | Moved to top-level |
| `resources/catalog.py` | `FunctionCatalogService` | Moved to top-level |
| `resources/features.py` | `compute_function_features`, `FunctionAstLoadRequest`, `load_function_asts` | Moved to top-level |
| `resources/graphs.py` | `GraphRuntimeOptions`, `build_graph_runtime` | Moved to top-level |

### Phase 9: Recipe Executor (4 errors) ✅

| File | Import |
|------|--------|
| `recipes/executor.py` | `AnalyticsContextConfig`, `AnalyticsContextProvider`, `CatalogProvider`, `GraphProvider` |

**Fix applied:** Moved all imports to top-level. No circular dependency issues.

### Phase 10: Semantic Roles Adapter (2 errors) ✅

| File | Import |
|------|--------|
| `adapters/semantic_roles.py` | `run_batch` (2 locations) |

**Fix applied:** Moved `from codeintel.ingestion.common import run_batch` to top-level.

---

## Final Error Count Summary

| Category | Original | Fixed |
|----------|----------|-------|
| Standard library imports | 2 | 2 ✅ |
| core/base.py provider imports | 7 | 7 ✅ |
| context.py provider imports | 5 | 5 ✅ |
| Plugin compute() imports | 17 | 17 ✅ |
| NetworkX imports | 3 | 3 ✅ |
| Resource provider imports | 7 | 7 ✅ |
| Recipe executor imports | 4 | 4 ✅ |
| Adapter imports | 2 | 2 ✅ |
| **Total** | **47** | **47 ✅** |

---

## Architectural Decisions

### String-Based Provider Lookup

The key architectural enhancement was adding string-based provider lookup to `ResourceRegistry`:

```python
# Without string lookup (causes circular import):
from codeintel.analytics.resources.catalog import CatalogProvider
provider = ctx.require(CatalogProvider)

# With string lookup (breaks the cycle):
provider: CatalogProvider = ctx.require_by_name("CatalogProvider")
```

This allows imports to be moved to `TYPE_CHECKING` blocks where they're only used for type annotations, not runtime execution.

### Moving Imports to Top-Level

For most imports in resource providers, recipe executor, and adapters, analysis revealed **no circular dependencies**. The imports were simply moved to the top of the file, which is the cleanest solution.

### No Suppressions Needed

The original plan included a fallback option of targeted per-file ignores. This was **not needed** - all 47 errors were resolved through proper import restructuring.

---

## Verification

```bash
# All PLC0415 errors resolved
$ uv run ruff check src/codeintel/analytics/ --select=PLC0415
All checks passed!

# All Ruff checks pass
$ uv run ruff check src/codeintel/analytics/
All checks passed!
```

### Summary

- **47 PLC0415 errors** identified across 27 files
- **47 errors resolved** without any suppressions
- Key techniques:
  1. TYPE_CHECKING blocks with string-based provider lookups (for circular dependency prevention)
  2. Moving imports to top-level (for most cases where no circular dependencies existed)
  3. String-based `require_by_name()` API added to `ResourceRegistry` and `PluginExecutionContext`

