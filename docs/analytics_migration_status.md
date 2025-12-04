# Analytics Unified Architecture Migration Status

## Executive Summary

This document summarizes the work completed to align the analytics subsystem with the unified infrastructure in `core/`, and outlines the remaining tasks needed to complete the migration.

---

## Background: Previous Architecture

### The Dual-System Problem

The CodeIntel codebase evolved with two parallel plugin/computation systems:

1. **Graphs Subsystem** (`src/codeintel/graphs/`)
   - Purpose: Build and analyze code structure graphs (call graphs, import graphs, CFG/DFG)
   - Plugin runtime with `GraphPluginProtocol`, `GraphPluginMetadata`
   - Recipe DSL with `GraphRecipe`, `GraphStage`
   - Resource providers with `ResourceContainer`
   - Compute layer for centrality, structural, and community metrics

2. **Analytics Subsystem** (`src/codeintel/analytics/`)
   - Purpose: Compute derived metrics, coverage analysis, risk factors
   - Separate plugin runtime with `AnalyticsPluginProtocol`, different metadata fields
   - Separate recipe DSL with `AnalyticsRecipe`
   - Separate resource registry with different registration patterns
   - Duplicate graph metric computations in `graph_metrics/metrics.py`

### Architectural Issues

| Issue | Description |
|-------|-------------|
| **Code Duplication** | 1000+ lines of graph metric code duplicated between systems |
| **Protocol Divergence** | `PluginMetadata` had different field names (`capabilities_provided` vs `provides`) |
| **Type Fragmentation** | Different types for same concepts (`GraphRecipe` vs `AnalyticsRecipe`) |
| **Resource Protocol Mismatch** | Graphs used `RESOURCE_NAME` ClassVar, analytics used `resource_name` property |
| **Context Isolation** | Each system had its own `PluginExecutionContext` with incompatible interfaces |
| **Maintenance Burden** | Bug fixes and improvements needed in two places |

### Directory Structure (Before)

```
src/codeintel/
├── graphs/
│   ├── core/
│   │   ├── protocol.py      # GraphPluginProtocol, GraphPluginMetadata
│   │   ├── context.py       # GraphExecutionContext
│   │   └── result.py        # GraphPluginResult
│   ├── recipes/
│   │   ├── dsl.py           # GraphRecipe, GraphStage
│   │   └── executor.py
│   ├── resources/
│   │   ├── protocol.py      # ResourceProvider (RESOURCE_NAME)
│   │   └── container.py
│   └── compute/
│       └── metrics/         # Centrality, structural metrics
│
├── analytics/
│   ├── core/
│   │   ├── plugin_protocol.py  # AnalyticsPluginProtocol
│   │   ├── execution_context.py
│   │   └── base.py             # capabilities_provided/required
│   ├── recipes/
│   │   ├── model.py            # AnalyticsRecipe
│   │   └── executor.py
│   ├── resources/
│   │   ├── protocol.py         # LazyResource (resource_name property)
│   │   └── registry.py
│   └── graph_metrics/
│       └── metrics.py          # Duplicated graph computations
```

---

## Why We Are Making Changes

### Goals

1. **Single Source of Truth**: One canonical definition for plugins, recipes, and resources
2. **Eliminate Duplication**: Share graph computation code between systems
3. **Consistent Types**: Same `PluginMetadata`, `Recipe`, `ResourceProvider` everywhere
4. **Composability**: Analytics can use graph plugins, graphs can use analytics data
5. **Maintainability**: Fix bugs once, improve once, test once

### Design Principles

- **Unified Core Package**: New `src/codeintel/core/` contains canonical definitions
- **Inheritance for Extension**: Domain-specific contexts extend core contexts
- **Re-export for Compatibility**: Subsystem modules re-export from core
- **No Backward-Compatible Shims**: Full migration, no legacy code paths

### Target Architecture

```
src/codeintel/
├── core/                      # NEW: Unified infrastructure
│   ├── plugins/
│   │   ├── protocol.py        # PluginProtocol, PluginMetadata, ValidationResult
│   │   ├── context.py         # PluginExecutionContext, PluginScratch
│   │   └── result.py          # PluginResult, PluginExecutionRecord
│   ├── recipes/
│   │   ├── model.py           # Recipe, RecipeStage, RecipeOptions
│   │   └── dsl.py             # RecipeBuilder, stage(), recipe()
│   └── resources/
│       ├── protocol.py        # ResourceProvider, ResourceProviderBase
│       └── registry.py        # ResourceRegistry
│
├── graphs/
│   ├── core/
│   │   ├── protocol.py        # GraphPluginMetadata extends PluginMetadata
│   │   └── context.py         # GraphPluginExecutionContext extends core
│   ├── recipes/
│   │   └── dsl.py             # Re-exports core + graph_stage(), graph_recipe()
│   └── resources/
│       └── protocol.py        # Re-exports core types
│
├── analytics/
│   ├── core/
│   │   ├── plugin_protocol.py # Re-exports core types
│   │   └── execution_context.py # AnalyticsPluginExecutionContext extends core
│   ├── recipes/
│   │   └── model.py           # Re-exports core types
│   └── resources/
│       └── protocol.py        # Re-exports core types
```

---

## Migration Scope

### Phase A: Graphs Subsystem Migration (COMPLETED)

The graphs subsystem was migrated first as a reference implementation.

#### A.1 Unified Core Infrastructure Created

| New File | Purpose |
|----------|---------|
| `core/plugins/protocol.py` | `PluginProtocol`, `PluginMetadata`, `ValidationResult`, `PluginResourceHints` |
| `core/plugins/context.py` | `PluginExecutionContext`, `PluginExecutionContextBuilder`, `PluginScratch` |
| `core/plugins/result.py` | `PluginResult`, `PluginExecutionRecord`, `PluginStatus` |
| `core/recipes/model.py` | `Recipe`, `RecipeStage`, `RecipeOptions`, `RecipeScope` |
| `core/recipes/dsl.py` | `RecipeBuilder`, `stage()`, `recipe()` |
| `core/resources/protocol.py` | `ResourceProvider`, `ResourceProviderBase` |
| `core/resources/registry.py` | `ResourceRegistry`, `ResourceNotFoundError` |

#### A.2 Graph Protocol Updates

| Change | Details |
|--------|---------|
| `GraphPluginMetadata` | Now extends `PluginMetadata` with graph-specific fields (`produces_graph_kinds`, `requires_graph_kinds`) |
| `GraphPluginExecutionContext` | Extends `PluginExecutionContext` with `graph_resources`, `require_graphs()` |
| Recipe DSL | `graph_recipe()` and `graph_stage()` wrap core `recipe()` and `stage()` |
| Resource Providers | All use `RESOURCE_NAME` ClassVar pattern |

#### A.3 Graph Files Updated

- **10 plugin files** in `graphs/plugins/`
- **4 runtime files** in `graphs/runtime/`
- **2 recipe files** in `graphs/recipes/`
- **40+ test files** in `tests/graphs/`

#### A.4 Legacy Code Removed from Graphs

| Removed | Replacement |
|---------|-------------|
| `GraphPluginResult` alias | Use `PluginResult` |
| `GraphExecutionContext` alias | Use `GraphPluginExecutionContext` |
| `GraphRecipe` type | Use `Recipe` |
| `GraphStage` type | Use `RecipeStage` |
| `BaseResourceProvider` | Use `ResourceProviderBase` |
| `graphs/core/result.py` | Deleted (use `core/plugins/result.py`) |

#### A.5 Graphs Validation Status

```
✓ ruff format: Pass
✓ ruff check: Pass  
✓ pyright: 0 errors
✓ pyrefly: 0 errors
✓ pytest tests/graphs/: 738 tests, 0 failures
```

### Phase B: Analytics Subsystem Migration (IN PROGRESS)

The analytics subsystem follows the same patterns established in graphs.

#### B.1 Completed Work

| Component | Status | Changes |
|-----------|--------|---------|
| Plugin Metadata Fields | ✓ Complete | `capabilities_provided` → `provides`, added `kind` field |
| Builders | ✓ Complete | Updated to construct unified `PluginMetadata` |
| Registry | ✓ Complete | Uses `provides`/`requires` for capability tracking |
| 16 Plugin Files | ✓ Complete | Migrated to string-based capability declarations |
| Resource Providers | Partial | Added `RESOURCE_NAME` to 4 providers |

#### B.2 Remaining Work

| Component | Status | Required Changes |
|-----------|--------|------------------|
| `LazyResource` Protocol | Pending | Add `RESOURCE_NAME` ClassVar, bridge with `resource_name` property |
| `ResourceRegistry` | Pending | Update to use `RESOURCE_NAME` pattern |
| `PluginExecutionContext` | Pending | Create `AnalyticsPluginExecutionContext` extending core |
| Recipe System | Pending | Re-export from `core/recipes`, alias `AnalyticsRecipe` |
| Graph Metrics Façade | Deferred | Convert to use `graphs/compute/metrics` (large refactoring) |

---

## Part 1: Detailed Changes (This Session)

### 1.1 Plugin Metadata Compatibility (COMPLETED)

The unified `PluginMetadata` in `core/plugins/protocol.py` uses different field names than the analytics code was using. All changes have been applied to align analytics with the unified schema.

#### Updated `analytics/core/builders.py`

**Changes Made:**
- Added `PluginKind` and `PluginIsolation` imports
- Changed `PluginContractsSection` fields:
  - `capabilities_provided: list[PluginCapability]` → `provides: list[str]`
  - `capabilities_required: list[PluginCapability]` → `requires: list[str]`
- Added `kind: PluginKind = "analytics"` to `PluginMetaSection`
- Changed `isolation_kind` type from `Literal["process", "thread"] | None` to `PluginIsolation = "none"`
- Added `kind()` builder method for fluent configuration
- Updated `provides()` and `requires()` methods to extract string names from `PluginCapability` objects
- Updated `build()` method to use new field names (`provides`, `requires`, `kind`)

#### Verified `analytics/core/base.py`

**Status:** Already using correct unified fields
- Uses `plugin_kind: ClassVar[PluginKind] = "analytics"`
- Uses `provides: ClassVar[tuple[str, ...]] = ()`
- Uses `requires: ClassVar[tuple[str, ...]] = ()`
- Uses `isolation_kind: ClassVar[PluginIsolation] = "none"`
- `metadata` property constructs `PluginMetadata` with correct fields

#### Updated `analytics/core/registry.py`

**Changes Made:**
- Added `PluginKind` and `PluginIsolation` imports
- Updated capability indexing to use `meta.provides` (string tuple) instead of `meta.capabilities_provided` (PluginCapability objects)
- Updated unregister method similarly
- Updated `_resolve_dependencies()` to use `plugin.metadata.requires` instead of `capabilities_required`
- Updated `_build_capability_index()` to use `plugin.metadata.provides`
- Updated `PluginMetaOptions` class:
  - Added `kind: PluginKind = "analytics"` field
  - Changed `isolation_kind` type to `PluginIsolation = "none"`
- Updated `PluginMetaOptionsInput` TypedDict with same changes
- Updated `to_metadata()` method:
  - Changed `_normalize_capability()` to return `str` instead of `PluginCapability`
  - Added `kind=self.kind` to `PluginMetadata` constructor
  - Changed `capabilities_provided` → `provides`
  - Changed `capabilities_required` → `requires`

#### Updated 16 Plugin Files

**Pattern Applied (each file):**
```python
# From:
capabilities_provided=(PluginCapability(name="x", kind="dataset"),)
capabilities_required=(PluginCapability(name="y", kind="dataset"),)

# To:
provides=("x",)
requires=("y",)
```

**Files Updated:**
| Directory | Files |
|-----------|-------|
| `functions/` | ast_features.py, history.py, effects.py, contracts.py |
| `coverage/` | test_edges.py |
| `data_models/` | build.py, usage.py |
| `dependencies/` | external.py |
| `semantic_roles/` | compute.py |
| `subsystems/` | build.py |
| `profiles/` | build.py |
| `config_data_flow/` | compute.py |
| `risk/` | factors.py |
| `history/` | timeseries.py |
| `tests/` | behavioral_coverage.py, profile.py |

**Additional Changes:**
- Removed unused `PluginCapability` imports (auto-fixed by Ruff)
- Added `kind="analytics"` to all `PluginMetadata` constructors

#### Updated `analytics/graphs/catalog.py`

**Changes Made:**
- Changed `meta.capabilities_provided` access to `meta.provides`
- Changed `meta.capabilities_required` access to `meta.requires`

### 1.2 Resource Provider Protocol Fixes (PARTIAL)

**Changes Made:**
- Added `RESOURCE_NAME: ClassVar[str]` to:
  - `AstProvider` in `analytics/resources/asts.py`
  - `FeaturesProvider` in `analytics/resources/features.py`
  - `GraphProvider` in `analytics/resources/graphs.py` (from previous session)
  - `CatalogProvider` in `analytics/resources/catalog.py` (from previous session)

---

## Part 2: Remaining Work

### Issue 1: Resource Provider Protocol Mismatch (CRITICAL)

**Problem:** The analytics `LazyResource` base class uses `resource_name` as an instance property, but the unified `core/resources/protocol.py` defines `ResourceProvider` with `RESOURCE_NAME` as a ClassVar.

**Files Affected:**
- `analytics/resources/protocol.py` - Contains `LazyResource` with `resource_name` property
- `analytics/resources/registry.py` - Accesses `resource_name` on providers
- `analytics/resources/module_map.py` - `ModuleMapProvider` needs `RESOURCE_NAME`

**Current Errors:**
```
analytics/resources/registry.py:96 - Cannot access attribute "resource_name" 
analytics/resources/registry.py:142 - Cannot access attribute "resource_name"
analytics/resources/registry.py:143 - Cannot access attribute "resource_name"
analytics/resources/registry.py:309 - Cannot access attribute "get_or_none"
analytics/resources/module_map.py:43 - "ClassVar" is not defined
analytics/resources/factory.py:158 - "ModuleMapProvider" incompatible with ResourceProvider
```

**Required Actions:**
1. Fix `ClassVar` import in `module_map.py`
2. Decide on unified approach for `LazyResource`:
   - Option A: Add `RESOURCE_NAME` ClassVar to `LazyResource` and all subclasses
   - Option B: Update analytics `ResourceRegistry` to use both `RESOURCE_NAME` and `resource_name`
   - Option C: Migrate all analytics providers to extend `core/resources/ResourceProviderBase`

### Issue 2: Analytics ResourceRegistry vs Core ResourceRegistry

**Problem:** Analytics has its own `ResourceRegistry` in `analytics/resources/registry.py` that differs from `core/resources/registry.py`:
- Analytics version uses `resource_type` (class) + `provider` pattern
- Analytics version has `get_or_none()` method expectations
- Core version uses `provider.RESOURCE_NAME` for registration

**Required Actions:**
1. Compare both implementations
2. Either:
   - Update analytics `ResourceRegistry` to match core patterns
   - Or keep analytics-specific registry and ensure protocol compatibility

### Issue 3: Graph Metrics Façade (DEFERRED)

**Status:** The plan indicated converting `analytics/graph_metrics/metrics.py` to use `graphs/compute/metrics/`, but this was marked as completed in a previous session.

**Current State:**
- `analytics/graph_metrics/metrics.py` (1115 lines) still has 92 direct NetworkX calls
- No imports from `codeintel.graphs.compute.metrics`
- Contains analytics-specific context management (`GraphContext`) that adds value

**Recommendation:** This is a larger refactoring that should be done incrementally:
1. Identify functions that purely duplicate `graphs/compute/metrics`
2. Replace those with calls to the unified compute functions
3. Keep analytics-specific adapter functions and data structures

### Issue 4: Execution Context Unification (NOT STARTED)

**Current State:**
- Analytics has `analytics/core/execution_context.py` with its own `PluginExecutionContext`
- Core has unified `core/plugins/context.py` with `PluginExecutionContext`
- Graphs uses `GraphPluginExecutionContext` extending the core context

**Required Actions:**
1. Create `AnalyticsPluginExecutionContext` extending `core/plugins/context.PluginExecutionContext`
2. Add analytics-specific fields (scope, analytics_resources)
3. Update all plugins to use the new context type

### Issue 5: Recipe System Unification (NOT STARTED)

**Current State:**
- Analytics has `analytics/recipes/model.py` with `AnalyticsRecipe`
- Core has unified `core/recipes/model.py` with `Recipe`
- Graphs already uses `core/recipes`

**Required Actions:**
1. Update `analytics/recipes/model.py` to import from `core/recipes`
2. Make `AnalyticsRecipe` an alias or thin wrapper if needed
3. Update `analytics/recipes/executor.py` to use unified types

---

## Part 3: Detailed Implementation Plan

### Phase 2: Fix Resource Provider Issues

#### Step 2.1: Fix Immediate Errors

```bash
# Files to modify:
src/codeintel/analytics/resources/module_map.py  # Add ClassVar import
src/codeintel/analytics/resources/protocol.py    # Add RESOURCE_NAME to LazyResource
src/codeintel/analytics/resources/registry.py    # Update to use RESOURCE_NAME
```

**Changes:**

1. **`module_map.py`**: Add `ClassVar` import
```python
from typing import TYPE_CHECKING, ClassVar
```

2. **`protocol.py`**: Add `RESOURCE_NAME` to `LazyResource`
```python
class LazyResource[T](ABC):
    RESOURCE_NAME: ClassVar[str] = ""  # Override in subclasses
    
    @property
    def resource_name(self) -> str:
        """Return the resource name (backward compatibility)."""
        return self.RESOURCE_NAME or self._name
```

3. **`registry.py`**: Update to check both `RESOURCE_NAME` and `resource_name`
```python
def _get_provider_name(self, provider: ResourceProvider[Any]) -> str:
    # Check ClassVar first (unified protocol)
    if hasattr(provider, "RESOURCE_NAME") and provider.RESOURCE_NAME:
        return provider.RESOURCE_NAME
    # Fall back to instance property (analytics LazyResource)
    if hasattr(provider, "resource_name"):
        return provider.resource_name
    return provider.__class__.__name__
```

#### Step 2.2: Add RESOURCE_NAME to All LazyResource Subclasses

**Files to update:**
- `analytics/resources/asts.py` ✓ (already done)
- `analytics/resources/features.py` ✓ (already done)
- `analytics/resources/graphs.py` ✓ (already done)
- `analytics/resources/catalog.py` ✓ (already done)
- `analytics/resources/module_map.py` (partial - needs ClassVar import)

### Phase 3: Unify Execution Context

#### Step 3.1: Create AnalyticsPluginExecutionContext

**File:** `src/codeintel/analytics/core/execution_context.py`

```python
from codeintel.core.plugins.context import (
    PluginExecutionContext as CorePluginExecutionContext,
    PluginExecutionContextBuilder as CorePluginExecutionContextBuilder,
)

@dataclass
class AnalyticsPluginExecutionContext(CorePluginExecutionContext):
    """Execution context for analytics plugins."""
    
    scope: AnalyticsScope | None = None
    analytics_resources: ResourceRegistry = field(default_factory=ResourceRegistry)
    
    def require_analytics[T](self, resource_type: type[T]) -> T:
        """Get an analytics-specific resource."""
        return self.analytics_resources.require(resource_type)
```

#### Step 3.2: Update Plugin Signatures

Update all analytics plugins to accept the new context type in their `execute()` and `validate_inputs()` methods.

### Phase 4: Unify Recipe System

#### Step 4.1: Update Recipe Model

**File:** `src/codeintel/analytics/recipes/model.py`

```python
from codeintel.core.recipes import Recipe, RecipeStage, RecipeOptions

# Alias for backward compatibility
AnalyticsRecipe = Recipe
```

#### Step 4.2: Update Recipe Executor

**File:** `src/codeintel/analytics/recipes/executor.py`

Update to use unified `Recipe` and `RecipeStage` types.

### Phase 5: Final Validation

```bash
# Run full validation suite
uv run ruff format src/codeintel/analytics/
uv run ruff check --fix src/codeintel/analytics/
uv run pyright --warnings --pythonversion=3.13 src/codeintel/analytics/
uv run pyrefly check src/codeintel/analytics/
uv run pytest tests/analytics/ -q
```

---

## Part 4: Files Summary

### Files Modified in This Session

| File | Changes |
|------|---------|
| `analytics/core/builders.py` | Updated to use unified field names, added `kind` |
| `analytics/core/registry.py` | Updated capability access patterns, added unified types |
| `analytics/core/plugins/functions/ast_features.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/functions/history.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/functions/effects.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/functions/contracts.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/coverage/test_edges.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/data_models/build.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/data_models/usage.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/dependencies/external.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/semantic_roles/compute.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/subsystems/build.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/profiles/build.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/config_data_flow/compute.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/risk/factors.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/history/timeseries.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/tests/behavioral_coverage.py` | Migrated to string-based provides/requires |
| `analytics/core/plugins/tests/profile.py` | Migrated to string-based provides/requires |
| `analytics/graphs/catalog.py` | Updated capability access to use provides/requires |
| `analytics/resources/asts.py` | Added RESOURCE_NAME ClassVar |
| `analytics/resources/features.py` | Added RESOURCE_NAME ClassVar |
| `analytics/resources/module_map.py` | Added RESOURCE_NAME (needs ClassVar import fix) |

### Files Still Requiring Changes

| File | Required Changes |
|------|-----------------|
| `analytics/resources/module_map.py` | Add ClassVar import |
| `analytics/resources/protocol.py` | Add RESOURCE_NAME to LazyResource |
| `analytics/resources/registry.py` | Update to use RESOURCE_NAME pattern |
| `analytics/core/execution_context.py` | Extend core PluginExecutionContext |
| `analytics/recipes/model.py` | Import from core/recipes |
| `analytics/recipes/executor.py` | Use unified recipe types |

---

## Validation Status

| Check | Status |
|-------|--------|
| `ruff format` | ✓ Pass |
| `ruff check` | ✓ Pass (analytics/core/) |
| `pyright` | ✗ 6 errors in analytics/resources/ |
| `pyrefly` | ✓ Pass (analytics/core/) |
| `pytest` | Not yet run |

---

## Estimated Remaining Effort

| Phase | Effort | Priority |
|-------|--------|----------|
| Fix Resource Provider Issues | 1-2 hours | HIGH |
| Unify Execution Context | 2-3 hours | MEDIUM |
| Unify Recipe System | 1 hour | MEDIUM |
| Graph Metrics Façade | 4-6 hours | LOW |
| Final Validation & Testing | 2 hours | HIGH |

**Total Estimated:** 10-14 hours

---

*Document generated: Session summary for analytics migration work*
