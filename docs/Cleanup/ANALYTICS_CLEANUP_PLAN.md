# Analytics Package Cleanup Plan

> **Generated:** 2025-12-13  
> **Updated:** 2025-12-13 (Phases 1-3 completed, consolidation opportunities added)  
> **Package:** `codeintel.analytics`  
> **Status:** Phases 1-3 Complete, Phase 5 Ready for Review

## Executive Summary

The `analytics` package cleanup has been completed through Phase 3:

**Completed:**
- ~~1 deprecated stub package removed (`adapters/`)~~ ✅
- ~~3 unused backward-compatibility aliases removed~~ ✅
- ~~1 unused protocol removed (`GraphRuntimePort`)~~ ✅
- ~~1 empty test directory deleted~~ ✅
- ~~Entire `ports/` package removed (unused re-exports)~~ ✅

**Remaining Consolidation Opportunities (Phase 5+):**
- Duplicate constants across graph metrics modules
- Duplicate lazy loading patterns in `__init__.py` files
- Similar orchestration patterns in graph metrics modules
- Re-export consolidation in `resources/protocol.py`

---

## Table of Contents

1. [Completed Work](#1-completed-work)
2. [Orphaned Modules Assessment](#2-orphaned-modules-assessment)
3. [Re-Export Consolidation](#3-re-export-consolidation)
4. [Active Modules Assessment](#4-active-modules-assessment)
5. [Consolidation Opportunities](#5-consolidation-opportunities)
6. [Implementation Checklist](#6-implementation-checklist)

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

---

## 2. Orphaned Modules Assessment

### 2.1 `analytics/graphs/plugin_catalog.py` - **NOT ORPHANED**

**Status:** ✅ Actively Used

Initial analysis incorrectly identified this as orphaned. It is used by:
- `scripts/render_graph_plugin_catalog.py`

**Recommendation:** Keep - used for documentation generation.

### 2.2 `analytics/graphs/contracts.py` - Test Infrastructure

**Status:** 🟢 Keep (Test-only)

**Purpose:** Contract checking helpers for graph metric plugins.

**Recommendation:** Keep for testing infrastructure. Not a cleanup candidate.

---

## 3. Re-Export Consolidation

### Status: 🟢 Lower Priority (Keep As-Is)

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

**Recommendation:** Keep - it's actively used and provides valid abstraction for analytics code. Consider consolidation in Phase 5 if resources layer is refactored.

---

## 4. Active Modules Assessment

### Heavily Used (Keep as-is)

| Module | Import Count | Notes |
|--------|-------------|-------|
| `analytics.runtime` | 16+ | Core graph runtime infrastructure |
| `analytics.parsing.ast_cache` | 16+ | Function AST caching |
| `analytics.compute.graphs` | 39+ | Graph algorithm primitives |
| `analytics.utilities.datasets` | 9+ | Dataset contracts |
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

---

## 5. Consolidation Opportunities

### 5.1 Duplicate Constants Across Graph Metrics Modules

**Status:** 🟡 Medium Priority

**Issue:** `CENTRALITY_SAMPLE_LIMIT = 500` is defined identically in two files:

| File | Line |
|------|------|
| `analytics/graphs/graph_metrics_ext.py` | 53 |
| `analytics/graphs/module_graph_metrics_ext.py` | 55 |

**Recommendation:** Extract to a shared constants module:

```python
# analytics/graphs/constants.py
CENTRALITY_SAMPLE_LIMIT = 500
EIGEN_MAX_ITER = 200
RICH_CLUB_PERCENTILE = 0.1
```

**Impact:** 2 files to update
**Risk:** None

---

### 5.2 Duplicate Lazy Loading Patterns

**Status:** 🟢 Lower Priority (Architectural)

**Issue:** Two `__init__.py` files implement nearly identical lazy loading patterns:

| File | Pattern |
|------|---------|
| `analytics/graphs/__init__.py` | `_LAZY_ATTRS` dict + `__getattr__` + `_wrap_lazy_attr` wrapper |
| `analytics/functions/__init__.py` | `_LAZY_ATTRS` dict + `_load` helper + `__getattr__` |

Both define:
- `_LAZY_ATTRS: dict[str, tuple[str, str]]` mapping names to (module, attr) tuples
- A `__getattr__` hook for lazy imports
- Wrapper functions for each exported callable

**Recommendation (Option A - Keep Status Quo):** The patterns are functionally correct and provide startup performance benefits. Leave as-is unless a unified lazy loading utility is developed.

**Recommendation (Option B - Consolidate):** Create a shared utility in `analytics/utilities/lazy_imports.py`:

```python
def lazy_module_attr(lazy_attrs: dict[str, tuple[str, str]]) -> Callable[[str], object]:
    """Create a __getattr__ hook for lazy module attribute loading."""
    def __getattr__(name: str) -> object:
        if name not in lazy_attrs:
            raise AttributeError(...)
        module_path, attr_name = lazy_attrs[name]
        return getattr(importlib.import_module(module_path), attr_name)
    return __getattr__
```

**Impact:** 2 files, potential simplification
**Risk:** Low - internal refactoring only

---

### 5.3 Similar Graph Metrics Orchestration Modules

**Status:** 🟢 Lower Priority (Architectural)

**Issue:** `graph_metrics_ext.py` and `module_graph_metrics_ext.py` have nearly identical structure:

| Aspect | `graph_metrics_ext.py` | `module_graph_metrics_ext.py` |
|--------|------------------------|-------------------------------|
| Lines | ~236 | ~214 |
| Main dataclass | `GraphViews` | `ImportGraphViews` |
| Compute function | `compute_graph_metrics_functions_ext` | `compute_graph_metrics_modules_ext` |
| Row builder | `build_function_metric_ext_rows` | `build_module_metric_ext_rows` |
| Graph source | Call graph | Import graph |

Both follow the same pattern:
1. Build graph views (undirected, reversed, etc.)
2. Compute centrality metrics
3. Compute structural metrics
4. Build rows and persist

**Recommendation:** Consider creating a generic `ExtendedMetricsOrchestrator` class that both can use, parameterized by graph type and row builder. This is an optional refactoring that would reduce code duplication by ~100 lines.

**Impact:** 2 files, medium refactoring effort
**Risk:** Medium - requires careful testing

---

### 5.4 Layered Architecture is Well-Designed

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

### 5.5 Subsystem Module Organization

**Status:** 🟢 Lower Priority

**Observation:** The `analytics/subsystems/` package has 4 modules:
- `affinity.py` - Subsystem affinity scoring
- `edge_stats.py` - Edge statistics
- `materialize.py` - Subsystem materialization
- `risk.py` - Risk computation

These are called from:
- `analytics/graphs/__init__.py` exports `build_subsystems`
- `build/plugins/analytics/subsystems/build.py`
- `build/plugins/analytics/risk/factors.py`

**Recommendation:** No consolidation needed. The subsystem modules are appropriately scoped.

---

## 6. Implementation Checklist

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

### Remaining Phases

#### Phase 5: Constants Consolidation (Optional)
- [ ] Create `analytics/graphs/constants.py` with shared constants
- [ ] Update `graph_metrics_ext.py` to import from constants
- [ ] Update `module_graph_metrics_ext.py` to import from constants
- [ ] Run tests to verify

#### Phase 6: Lazy Loading Consolidation (Optional)
- [ ] Evaluate whether unified lazy loading utility is worth the effort
- [ ] If yes, create `analytics/utilities/lazy_imports.py`
- [ ] Update `analytics/graphs/__init__.py` to use utility
- [ ] Update `analytics/functions/__init__.py` to use utility

#### Phase 7: Orchestration Consolidation (Optional)
- [ ] Design `ExtendedMetricsOrchestrator` abstraction
- [ ] Refactor `graph_metrics_ext.py` to use orchestrator
- [ ] Refactor `module_graph_metrics_ext.py` to use orchestrator
- [ ] Run full test suite

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
| Duplicate constants | - | 1 (CENTRALITY_SAMPLE_LIMIT) |
| Lazy loading patterns | - | 2 similar patterns |

---

## Related Documents

- [GRAPHS_CLEANUP_PLAN.md](./GRAPHS_CLEANUP_PLAN.md)
- [BUILD_CLEANUP_PLAN.md](./BUILD_CLEANUP_PLAN.md)
- [BUILD_CONSOLIDATION_PLAN.md](./BUILD_CONSOLIDATION_PLAN.md)
- [BUILD_REFINEMENT_PLAN.md](./BUILD_REFINEMENT_PLAN.md)
