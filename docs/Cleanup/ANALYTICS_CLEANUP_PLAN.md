# Analytics Package Cleanup Plan

> **Generated:** 2025-12-13  
> **Updated:** 2025-12-13 (All phases completed, new opportunities identified)  
> **Package:** `codeintel.analytics`  
> **Status:** Core Phases Complete ✅ | Future Opportunities Identified

## Executive Summary

The `analytics` package cleanup is fully complete through all core phases:

**Completed:**
- ~~1 deprecated stub package removed (`adapters/`)~~ ✅
- ~~3 unused backward-compatibility aliases removed~~ ✅
- ~~1 unused protocol removed (`GraphRuntimePort`)~~ ✅
- ~~1 empty test directory deleted~~ ✅
- ~~Entire `ports/` package removed (unused re-exports)~~ ✅
- ~~Constants consolidated into shared module~~ ✅
- ~~Lazy loading utility created and applied~~ ✅
- ~~Generic orchestrator for extended metrics~~ ✅
- ~~History module lazy loading adoption~~ ✅

**Summary of Cleanup Impact:**
- 3 new reusable modules created
- ~100 lines of duplicate code eliminated
- Cleaner separation of concerns
- All public APIs preserved

**Newly Identified Opportunities (Future Phases):**
- Additional constants scattered across modules
- Inconsistent persistence patterns (2 different approaches)
- Symbol graph metrics duplication (similar to orchestrator pattern)

---

## Table of Contents

1. [Completed Work](#1-completed-work)
2. [Orphaned Modules Assessment](#2-orphaned-modules-assessment)
3. [Re-Export Consolidation](#3-re-export-consolidation)
4. [Active Modules Assessment](#4-active-modules-assessment)
5. [Consolidation Opportunities (Completed)](#5-consolidation-opportunities-completed)
6. [Future Consolidation Opportunities](#6-future-consolidation-opportunities)
7. [Implementation Checklist](#7-implementation-checklist)

---

## 1. Completed Work

### Phase 1: Immediate Cleanup ✅

**Completed 2025-12-13**

- Deleted deprecated `src/codeintel/analytics/adapters/` directory
- Removed 3 unused backward-compatibility aliases from `analytics/data_models/core.py`:
  - `_safe_unparse = safe_unparse`
  - `_literal_value = literal_value`
  - `_call_name = call_name`
- Deleted empty `tests/analytics/ports/` directory

### Phase 2: Protocol Cleanup ✅

**Completed 2025-12-13**

- Deleted `src/codeintel/analytics/ports/graphs.py` (contained unused `GraphRuntimePort` protocol)
- Updated `analytics/ports/__init__.py` to remove `GraphRuntimePort` export

### Phase 3: Ports Package Removal ✅

**Completed 2025-12-13**

- Deleted entire `src/codeintel/analytics/ports/` directory
- Updated `tests/analytics/test_public_exports.py` to remove ports-related tests
- All files removed:
  - `__init__.py` (re-exports only)
  - `catalog.py` (re-export from `graphs.ports`)
  - `storage.py` (re-export from `graphs.ports`)
  - `graphs.py` (unused protocol)

### Phase 5: Constants Consolidation ✅

**Completed 2025-12-13**

Created shared constants module to eliminate duplication:

**New File:** `src/codeintel/analytics/graphs/constants.py`

```python
CENTRALITY_SAMPLE_LIMIT: int = 500
EIGEN_MAX_ITER: int = 200
RICH_CLUB_PERCENTILE: float = 0.1
```

**Updated Files:**
- `analytics/graphs/graph_metrics_ext.py` - imports `CENTRALITY_SAMPLE_LIMIT`, `EIGEN_MAX_ITER`
- `analytics/graphs/module_graph_metrics_ext.py` - imports `CENTRALITY_SAMPLE_LIMIT`, `RICH_CLUB_PERCENTILE`

### Phase 6: Lazy Loading Consolidation ✅

**Completed 2025-12-13**

Created reusable lazy loading utility and refactored existing implementations:

**New File:** `src/codeintel/analytics/utilities/lazy_module.py`

Provides:
- `LazyAttrMap` - Type alias for lazy attribute mappings
- `make_lazy_getattr()` - Factory for creating `__getattr__` functions
- `lazy_callable()` - Factory for creating lazy-loading callable wrappers

**Refactored Files:**

| File | Before | After |
|------|--------|-------|
| `analytics/graphs/__init__.py` | 108 lines with inline `_wrap_lazy_attr` and `__getattr__` | 77 lines using `lazy_module` utilities |
| `analytics/functions/__init__.py` | Inline `_load` helper | Uses `make_lazy_getattr()` for fallback |

**Benefits:**
- Centralized lazy loading logic in one testable module
- Reduced code duplication by ~30 lines
- Consistent error messages across packages
- Optional caching in globals for performance

### Phase 7: Orchestrator Consolidation ✅

**Completed 2025-12-13**

Created a generic orchestration framework for extended graph metrics:

**New File:** `src/codeintel/analytics/graphs/orchestrator.py`

Provides:
- `GraphViews` - Shared dataclass for graph variants (graph, simple_graph, undirected)
- `ExtendedMetricsConfig[TSlices, TRow]` - Generic configuration for metrics orchestration
- `ExtendedMetricsRequest` - Request parameters bundle
- `build_graph_views()` - Standard graph view builder
- `compute_extended_metrics()` - Generic orchestration function

**Refactored Files:**

| File | Before | After | Savings |
|------|--------|-------|---------|
| `analytics/graphs/graph_metrics_ext.py` | 237 lines | ~150 lines | ~87 lines |
| `analytics/graphs/module_graph_metrics_ext.py` | 215 lines | ~145 lines | ~70 lines |

**Benefits:**
- Single orchestration implementation shared by both modules
- Type-safe configuration with PEP 695 generics
- Cleaner separation: orchestrator handles workflow, modules handle domain logic
- Easier to add new extended metrics types in the future

### Phase 8: History Lazy Loading Adoption ✅

**Completed 2025-12-13**

Refactored `analytics/history/__init__.py` to use the shared lazy loading utility:

**Before:** Inline `__getattr__` implementation (45 lines)
**After:** Uses `make_lazy_getattr()` (32 lines)

---

## 2. Orphaned Modules Assessment

### 2.1 `analytics/graphs/plugin_catalog.py` - **NOT ORPHANED**

**Status:** ✅ Actively Used

Initial analysis incorrectly identified this as orphaned. It is used by:
- `scripts/render_graph_plugin_catalog.py`

**Recommendation:** Keep - used for documentation generation.

### 2.2 `analytics/graphs/contracts.py` - Test Infrastructure

**Status:** ✅ Keep (Test-only)

**Purpose:** Contract checking helpers for graph metric plugins.

**Recommendation:** Keep for testing infrastructure. Not a cleanup candidate.

---

## 3. Re-Export Consolidation

### Status: ✅ Complete (Keep As-Is)

### 3.1 `analytics/resources/protocol.py`

**Current State:** Only re-exports from `codeintel.core.resources`:

```python
from codeintel.core.resources import (
    LazyResource,
    ResourceError,
    ResourceNotFoundError,
    ResourceNotLoadedError,
    ResourceProvider,
    ResourceProviderBase,
    ResourceRegistry,
)
```

**Evidence:** 7 files actively import from this module.

**Recommendation:** Keep - it's actively used and provides valid abstraction for analytics code.

---

## 4. Active Modules Assessment

### Heavily Used (Keep as-is)

| Module | Import Count | Notes |
|--------|-------------|-------|
| `analytics.runtime` | 16+ | Core graph runtime infrastructure |
| `analytics.parsing.ast_cache` | 16+ | Function AST caching |
| `analytics.compute.graphs` | 39+ | Graph algorithm primitives |
| `analytics.utilities.datasets` | 9+ | Dataset contracts |
| `analytics.utilities.lazy_module` | 3 | Lazy loading utility |
| `analytics.graphs.orchestrator` | 2 | Extended metrics orchestration |
| `analytics.ast_features` | 23+ | AST feature extraction |
| `analytics.parsing.*` | 37+ | Parsing infrastructure |
| `analytics.testing.*` | 30+ | Test analytics |
| `analytics.resources.*` | 24+ | Lazy resource providers |

### Moderately Used (Active)

| Module | Import Count | Notes |
|--------|-------------|-------|
| `analytics.utilities.persistence` | 7 | DeleteScope |
| `analytics.compute.row_builders` | 6+ | Row building utilities |
| `analytics.resources.features` | 5 | FeaturesProvider |
| `analytics.compute.evidence` | 6+ | EvidenceCollector |
| `analytics.graphs.constants` | 2 | Shared graph constants |

---

## 5. Consolidation Opportunities (Completed)

### 5.1 Duplicate Constants Across Graph Metrics Modules ✅

**Status:** ✅ Complete

See [Phase 5: Constants Consolidation](#phase-5-constants-consolidation-) above.

---

### 5.2 Duplicate Lazy Loading Patterns ✅

**Status:** ✅ Complete

See [Phase 6: Lazy Loading Consolidation](#phase-6-lazy-loading-consolidation-) above.

---

### 5.3 Similar Graph Metrics Orchestration Modules ✅

**Status:** ✅ Complete

See [Phase 7: Orchestrator Consolidation](#phase-7-orchestrator-consolidation-) above.

---

### 5.4 Minor Lazy Loading Adoption Opportunity ✅

**Status:** ✅ Complete

See [Phase 8: History Lazy Loading Adoption](#phase-8-history-lazy-loading-adoption-) above.

---

### 5.5 Layered Architecture is Well-Designed ✅

**Status:** ✅ No Action Needed

The current architecture follows a clean separation:

```
┌─────────────────────────────────────────────┐
│           analytics/graphs/                  │ ← High-level orchestration
│  (graph_metrics.py, symbol_graph_metrics.py) │   (I/O, persistence, runtime)
├─────────────────────────────────────────────┤
│        analytics/compute/graphs/             │ ← Pure computation primitives
│    (centrality.py, components.py, etc.)      │   (no I/O, stateless)
├─────────────────────────────────────────────┤
│        graphs/compute/metrics/               │ ← Core NetworkX wrappers
│  (centrality.py, structural.py, etc.)        │   (lowest-level algorithms)
└─────────────────────────────────────────────┘
```

This layering is correct and enables:
- Pure computation functions that are easy to test
- Separation of I/O concerns from algorithms
- Reuse of core metrics across different orchestration contexts

**Recommendation:** Maintain this architecture. It's well-designed.

---

### 5.6 Subsystem Module Organization ✅

**Status:** ✅ No Action Needed

The `analytics/subsystems/` package has 4 appropriately-scoped modules:
- `affinity.py` - Subsystem affinity scoring
- `edge_stats.py` - Edge statistics
- `materialize.py` - Subsystem materialization
- `risk.py` - Risk computation

No consolidation needed.

---

## 6. Future Consolidation Opportunities

The following opportunities were identified during the Phase 7-8 implementation and are candidates for future cleanup work.

### 6.1 Additional Constants Duplication

**Status:** 🟡 Medium Priority

**Issue:** Several constants are duplicated across multiple modules:

| Constant | Value | Locations |
|----------|-------|-----------|
| `MAX_BETWEENNESS_NODES` | 1000 | `symbol_graph_metrics.py:37`, `config_graph_metrics.py:41` |
| `MAX_COMMUNITY_NODES` | 5000 | `symbol_graph_metrics.py:38` |
| `MAX_CFG_CENTRALITY_SAMPLE` | 100 | `cfg_dfg/materialize.py:161` |
| `MAX_DFG_CENTRALITY_SAMPLE` | 100 | `cfg_dfg/materialize.py:163`, `cfg_dfg/dfg_core.py:21` |
| `MAX_CFG_EIGEN_SAMPLE` | 200 | `cfg_dfg/materialize.py:162` |

**Recommendation:** Extend `analytics/graphs/constants.py` (or create `analytics/compute/constants.py`) to include these constants:

```python
# Graph metrics sampling limits
MAX_BETWEENNESS_NODES: int = 1000
MAX_COMMUNITY_NODES: int = 5000

# CFG/DFG sampling limits
MAX_CFG_CENTRALITY_SAMPLE: int = 100
MAX_CFG_EIGEN_SAMPLE: int = 200
MAX_DFG_CENTRALITY_SAMPLE: int = 100
```

**Impact:** 5 files to update
**Risk:** None

---

### 6.2 Inconsistent Persistence Patterns

**Status:** 🟡 Medium Priority

**Issue:** Two different persistence patterns are used across graph metrics modules:

**Pattern A - Unified Dataset Contract (Preferred):**
```python
# Used in: orchestrator.py, graph_metrics.py
contract = get_analytics_dataset_contract(gateway, table_key)
validated_rows = validate_contract_rows(contract.table_key, rows)
insert_analytics_rows(gateway, contract, validated_rows, delete_scope=..., scope=...)
```

**Pattern B - Direct Ibis Write:**
```python
# Used in: symbol_graph_metrics.py, subsystem_graph_metrics.py, 
#          config_graph_metrics.py, config_data_flow.py, subsystem_agreement.py
contract = DATASET_CONTRACTS_BY_TABLE_KEY[table_key]
validated_rows = validate_tuple_rows(table_key, rows, schema=contract.schema)
backend.delete_for_snapshot(table_key, repo=repo, commit=commit)
gateway.ibis.write(table_key, validated_rows, columns=[...])
```

**Files using Pattern A:** 3 files (preferred, cleaner)
**Files using Pattern B:** 6 files (legacy, more verbose)

**Recommendation:** Migrate Pattern B files to use Pattern A for consistency:
- Reduces boilerplate (no manual column lists)
- Centralizes delete + insert logic
- Consistent error handling

**Impact:** 6 files to update
**Risk:** Low - same underlying behavior

---

### 6.3 Symbol Graph Metrics Duplication

**Status:** 🟢 Lower Priority

**Issue:** `compute_symbol_graph_metrics_modules` and `compute_symbol_graph_metrics_functions` (255 lines total) have ~80% identical structure:

| Aspect | `_modules` | `_functions` |
|--------|------------|--------------|
| Lines | ~106 | ~107 |
| Graph source | `ensure_symbol_module_graph()` | `ensure_symbol_function_graph()` |
| Filter source | ModuleRepository | FunctionRepository |
| Target table | `analytics.symbol_graph_metrics_modules` | `analytics.symbol_graph_metrics_functions` |
| Row builder | `build_symbol_module_rows` | `build_symbol_function_rows` |

Both follow identical patterns:
1. Resolve runtime options and snapshot
2. Ensure table exists
3. Resolve graph context
4. Get and filter graph
5. Compute undirected centralities
6. Compute structural metrics
7. Compute component IDs
8. Build and validate rows
9. Delete and write

**Recommendation:** Consider creating an "undirected metrics orchestrator" similar to `orchestrator.py`, but for undirected symbol graphs. This would require:
- A new `UndirectedMetricsConfig` dataclass
- A `compute_undirected_symbol_metrics()` orchestrator function
- Refactoring both functions to use the orchestrator

**Impact:** 2 files + 1 new orchestrator
**Savings:** ~50 lines
**Risk:** Medium - requires careful testing

---

### 6.4 Config Graph Metrics Structure

**Status:** 🟢 Lower Priority (Architectural)

**Observation:** `config_graph_metrics.py` (304 lines) handles 4 different projection types with repetitive patterns:
- Key projections
- Module projections
- Key-key edge metrics
- Module-module edge metrics

**Recommendation:** Consider a table-driven approach where projection configurations are defined declaratively. However, this is lower priority as the current code is functional and the projections have subtle differences.

---

## 7. Implementation Checklist

### Completed Phases ✅

#### Phase 1: Immediate Cleanup ✅
- [x] Delete `src/codeintel/analytics/adapters/` directory
- [x] Remove 3 aliases from `analytics/data_models/core.py`
- [x] Delete empty `tests/analytics/ports/` directory
- [x] Run tests to verify no regressions

#### Phase 2: Protocol Cleanup ✅
- [x] Delete `src/codeintel/analytics/ports/graphs.py`
- [x] Update `analytics/ports/__init__.py` to remove `GraphRuntimePort`
- [x] Run tests to verify no regressions

#### Phase 3: Ports Package Removal ✅
- [x] Verify no external packages depend on `analytics.ports`
- [x] Delete `src/codeintel/analytics/ports/` directory
- [x] Update `tests/analytics/test_public_exports.py` to remove ports tests
- [x] Run full test suite and type checking

#### Phase 4: Documentation ✅
- [x] Update this document with completion status
- [x] Correct assessment of `analytics/graphs/plugin_catalog.py` (not orphaned)
- [x] Add consolidation opportunities section

#### Phase 5: Constants Consolidation ✅
- [x] Create `analytics/graphs/constants.py` with shared constants
- [x] Update `graph_metrics_ext.py` to import from constants
- [x] Update `module_graph_metrics_ext.py` to import from constants
- [x] Run tests to verify

#### Phase 6: Lazy Loading Consolidation ✅
- [x] Create `analytics/utilities/lazy_module.py` with reusable utilities
- [x] Refactor `analytics/graphs/__init__.py` to use `lazy_callable` and `make_lazy_getattr`
- [x] Refactor `analytics/functions/__init__.py` to use `make_lazy_getattr`
- [x] Run full test suite and type checking

#### Phase 7: Orchestration Consolidation ✅
- [x] Create `analytics/graphs/orchestrator.py` with generic orchestration framework
- [x] Refactor `graph_metrics_ext.py` to use orchestrator
- [x] Refactor `module_graph_metrics_ext.py` to use orchestrator
- [x] Run full test suite

#### Phase 8: History Lazy Loading Adoption ✅
- [x] Refactor `analytics/history/__init__.py` to use `make_lazy_getattr`
- [x] Run full test suite and type checking

---

### Future Phases (Optional)

#### Phase 9: Extended Constants Consolidation
- [ ] Add `MAX_BETWEENNESS_NODES`, `MAX_COMMUNITY_NODES` to constants
- [ ] Add `MAX_CFG_CENTRALITY_SAMPLE`, `MAX_DFG_CENTRALITY_SAMPLE`, `MAX_CFG_EIGEN_SAMPLE`
- [ ] Update `symbol_graph_metrics.py` to import from constants
- [ ] Update `config_graph_metrics.py` to import from constants
- [ ] Update `cfg_dfg/materialize.py` to import from constants
- [ ] Update `cfg_dfg/dfg_core.py` to import from constants

#### Phase 10: Persistence Pattern Standardization
- [ ] Migrate `symbol_graph_metrics.py` to use `insert_analytics_rows()`
- [ ] Migrate `subsystem_graph_metrics.py` to use `insert_analytics_rows()`
- [ ] Migrate `config_graph_metrics.py` to use `insert_analytics_rows()`
- [ ] Migrate `config_data_flow.py` to use `insert_analytics_rows()`
- [ ] Migrate `subsystem_agreement.py` to use `insert_analytics_rows()`

#### Phase 11: Symbol Metrics Orchestrator
- [ ] Design undirected metrics orchestrator abstraction
- [ ] Create `analytics/graphs/symbol_orchestrator.py`
- [ ] Refactor `compute_symbol_graph_metrics_modules` to use orchestrator
- [ ] Refactor `compute_symbol_graph_metrics_functions` to use orchestrator

---

## New Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `src/codeintel/analytics/graphs/constants.py` | Shared graph metrics constants | 24 |
| `src/codeintel/analytics/utilities/lazy_module.py` | Reusable lazy loading utilities | 118 |
| `src/codeintel/analytics/graphs/orchestrator.py` | Generic extended metrics orchestrator | ~220 |

---

## Verification Commands

After implementing changes, run:

```bash
# Type checking
uv run pyright --warnings --pythonversion=3.13
uv run pyrefly check

# Linting
uv run ruff check --fix

# Full test suite
uv run pytest -q

# Verify no dead code introduced
uv run vulture src/codeintel/analytics --min-confidence 90
```

---

## Comparison with Graphs Package

| Item | Graphs Package | Analytics Package |
|------|----------------|-------------------|
| Empty directories | 2 (core/, runtime/) ✅ | 0 |
| Deprecated stubs | 1 (adapters) ✅ | 1 (adapters) ✅ |
| Unused aliases | 9 ✅ | 3 ✅ |
| Unused protocols | 1 (ParsingPort) ✅ | 1 (GraphRuntimePort) ✅ |
| Re-export packages | - | 1 (ports/) ✅ |
| Orphaned modules | - | 0 (plugin_catalog is used) |
| Empty test dirs | - | 1 (tests/analytics/ports/) ✅ |
| Duplicate constants | - | 1 set consolidated ✅, 1 set remaining |
| Lazy loading patterns | - | 3 → unified ✅ |
| Orchestration patterns | - | 2 → unified ✅ |
| Persistence patterns | - | 2 different patterns (standardization pending) |
| **New utilities created** | - | 3 (constants.py, lazy_module.py, orchestrator.py) |

---

## Related Documents

- [GRAPHS_CLEANUP_PLAN.md](./GRAPHS_CLEANUP_PLAN.md)
- [BUILD_CLEANUP_PLAN.md](./BUILD_CLEANUP_PLAN.md)
- [BUILD_CONSOLIDATION_PLAN.md](./BUILD_CONSOLIDATION_PLAN.md)
- [BUILD_REFINEMENT_PLAN.md](./BUILD_REFINEMENT_PLAN.md)
