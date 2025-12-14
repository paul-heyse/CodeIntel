# Analytics Package Cleanup Plan

> **Generated:** 2025-12-13  
> **Updated:** 2025-12-13 (Phases 9-10 completed; 7 new consolidation opportunities identified)  
> **Package:** `codeintel.analytics`  
> **Status:** Phase 1-10 Complete ✅ | Phase 11-18 Available for Implementation

## Executive Summary

The `analytics` package cleanup is complete through phases 1-10. Following the recent consolidation work, a deep analysis revealed 7 additional consolidation opportunities (phases 11-18).

**Completed (Phases 1-10):**
- ~~1 deprecated stub package removed (`adapters/`)~~ ✅
- ~~3 unused backward-compatibility aliases removed~~ ✅
- ~~1 unused protocol removed (`GraphRuntimePort`)~~ ✅
- ~~1 empty test directory deleted~~ ✅
- ~~Entire `ports/` package removed (unused re-exports)~~ ✅
- ~~Constants consolidated into shared module~~ ✅
- ~~Lazy loading utility created and applied~~ ✅
- ~~Generic orchestrator for extended metrics~~ ✅
- ~~History module lazy loading adoption~~ ✅
- ~~Extended constants consolidated (5 additional constants)~~ ✅
- ~~Persistence patterns standardized via `bulk_insert()`~~ ✅

**Completed Cleanup Impact:**
- 3 new reusable modules created
- ~200 lines of duplicate code eliminated
- Persistence standardized across 5 files
- Cleaner separation of concerns
- All public APIs preserved

**Remaining Opportunities (Newly Identified):**
- 🔴 Duplicate helper functions (`_to_records`, `_degree_dict`, etc.) - ~136 lines savings
- 🟡 Type conversion helper consolidation - ~15 lines savings
- 🟡 Profile writer pattern consolidation - ~40 lines savings
- 🟡 CFG/DFG shared helpers module - ~68 lines savings
- 🟠 Remaining persistence migration (`subsystems/materialize.py`)
- 🟢 Row builder input dataclass unification - ~22 lines savings
- 🟢 CFG/DFG context consolidation - ~20 lines savings
- 🟢 Symbol metrics orchestrator (Phase 11) - ~50 lines savings

**Total Potential Savings:** ~350+ lines of duplicate/boilerplate code

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

The following opportunities were identified during Phase 7-8 implementation. Sections 6.1 and 6.2 are now complete.

### 6.1 Additional Constants Duplication ✅

**Status:** ✅ Completed (Phase 9)

**Completed 2025-12-13**

All duplicate constants have been consolidated into `analytics/graphs/constants.py`:

| Constant | Value | Files Updated |
|----------|-------|---------------|
| `MAX_BETWEENNESS_NODES` | 1000 | `symbol_graph_metrics.py`, `config_graph_metrics.py` |
| `MAX_COMMUNITY_NODES` | 5000 | `symbol_graph_metrics.py` |
| `MAX_CFG_CENTRALITY_SAMPLE` | 100 | `cfg_dfg/materialize.py` |
| `MAX_DFG_CENTRALITY_SAMPLE` | 100 | `cfg_dfg/materialize.py`, `cfg_dfg/dfg_core.py` |
| `MAX_CFG_EIGEN_SAMPLE` | 200 | `cfg_dfg/materialize.py`, `cfg_dfg/dfg_core.py` |

---

### 6.2 Inconsistent Persistence Patterns ✅

**Status:** ✅ Completed (Phase 10)

**Completed 2025-12-13**

All files now use `DuckDBPolicyBackend.bulk_insert()` instead of manual `gateway.ibis.write()` with column lists:

**Standardized Pattern:**
```python
backend = DuckDBPolicyBackend(gateway)
backend.ensure_table(table_key)
# ... build rows ...
validated_rows = validate_tuple_rows(table_key, rows, schema=contract.schema)
backend.delete_for_snapshot(table_key, repo=repo, commit=commit)
backend.bulk_insert(table_key, validated_rows)  # Schema-derived columns
```

**Files Updated:**
- `subsystem_agreement.py`
- `subsystem_graph_metrics.py`
- `symbol_graph_metrics.py`
- `config_data_flow.py`
- `config_graph_metrics.py`

**Additional fix:** Added numpy type conversion in `validate_tuple_rows()` to ensure DuckDB compatibility.

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

### 6.5 Duplicate Helper Functions Across Modules 🔴

**Status:** 🟡 Medium Priority (Newly Identified)

**Issue:** Several helper functions are duplicated across multiple modules, violating DRY principles.

#### 6.5.1 `_to_records` (DataFrame → list[dict])

**Duplicated 3 times with identical implementation:**

| File | Lines |
|------|-------|
| `analytics/compute/row_builders/graph_metrics.py:27-35` | 9 |
| `analytics/functions/metrics.py:68-76` | 9 |
| `analytics/compute/functions/goids.py:115-123` | 9 |

**Implementation:**
```python
def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return cast("list[dict[str, Any]]", df.to_dict(orient="records"))
```

**Recommendation:** Consolidate into `analytics/utilities/dataframe.py` or add to existing `profiles/utils.py`.

**Impact:** 3 files
**Savings:** ~18 lines

---

#### 6.5.2 `_degree_dict` (NetworkX degree → dict)

**Duplicated 2 times with identical implementation:**

| File | Location |
|------|----------|
| `analytics/cfg_dfg/cfg_core.py:86-104` | 19 lines |
| `analytics/cfg_dfg/dfg_core.py:79-97` | 19 lines |

**Implementation:**
```python
def _degree_dict(
    graph: nx.DiGraph,
    *,
    direction: str,
    weight: str | None = None,
) -> dict[int, int]:
    raw_pairs = (
        graph.in_degree(weight=weight) if direction == "in" else graph.out_degree(weight=weight)
    )
    pairs = cast("Iterable[tuple[int, int | float]]", raw_pairs)
    return {int(node): int(deg) for node, deg in pairs}
```

**Recommendation:** Move to `analytics/compute/graphs/types.py` or create `analytics/cfg_dfg/helpers.py`.

**Impact:** 2 files
**Savings:** ~19 lines

---

#### 6.5.3 `parse_block_idx` / `_parse_block_idx`

**Duplicated 2 times with identical logic (different naming):**

| File | Function Name |
|------|---------------|
| `analytics/cfg_dfg/cfg_core.py:152-169` | `parse_block_idx` |
| `analytics/cfg_dfg/dfg_core.py:133-150` | `_parse_block_idx` |

**Implementation:**
```python
def parse_block_idx(block_id: str | int | None) -> int | None:
    if block_id is None:
        return None
    block_text = str(block_id)
    if "block" not in block_text:
        return None
    try:
        return int(block_text.rsplit("block", 1)[-1])
    except ValueError:
        return None
```

**Recommendation:** Move to `analytics/cfg_dfg/__init__.py` or `helpers.py`.

**Impact:** 2 files
**Savings:** ~17 lines

---

#### 6.5.4 `function_metadata` / `dfg_function_metadata`

**Duplicated 2 times with 100% identical implementations:**

| File | Function Name | Lines |
|------|---------------|-------|
| `analytics/cfg_dfg/cfg_core.py:420-451` | `function_metadata` | 32 |
| `analytics/cfg_dfg/dfg_core.py:335-366` | `dfg_function_metadata` | 32 |

Both execute the exact same SQL query and return the same mapping.

**Recommendation:** Consolidate into a single `load_function_metadata` function in `analytics/cfg_dfg/__init__.py`.

**Impact:** 2 files + 1 caller update (`materialize.py`)
**Savings:** ~32 lines

---

### 6.6 Type Conversion Helper Consolidation 🟡

**Status:** 🟡 Medium Priority (Newly Identified)

**Issue:** Similar type conversion helpers exist in different locations with slightly different signatures.

| Helper | Location | Signature |
|--------|----------|-----------|
| `_int_or_none` | `compute/row_builders/graph_metrics_ext.py:21-34` | `(float \| str \| Decimal \| None) -> int \| None` |
| `optional_int` | `profiles/utils.py:89-109` | `(object \| None) -> int \| None` |
| `int_or_default` | `profiles/utils.py:112-122` | `(object \| None, default: int) -> int` |
| `optional_str` | `profiles/utils.py:77-86` | `(object \| None) -> str \| None` |
| `optional_float` | `profiles/utils.py:125-145` | `(object \| None) -> float \| None` |
| `optional_bool` | `profiles/utils.py:148-169` | `(object \| None) -> bool \| None` |

**Recommendation:** Consolidate all type conversion helpers into a single location:
- Move to `analytics/utilities/type_coercion.py`
- Unify `_int_or_none` and `optional_int` (use the more flexible `optional_int`)
- Re-export from `analytics/utilities/__init__.py`

**Impact:** 2+ files
**Savings:** ~15 lines + improved consistency

---

### 6.7 Row Builder Input Dataclass Unification 🟢

**Status:** 🟢 Lower Priority (Newly Identified)

**Issue:** Several input dataclasses share nearly identical structures:

#### 6.7.1 Symbol Metric Inputs

| Dataclass | Location |
|-----------|----------|
| `SymbolModuleMetricInputs` | `compute/row_builders/symbol_metrics.py:18-28` |
| `SymbolFunctionMetricInputs` | `compute/row_builders/symbol_metrics.py:31-41` |

**Identical fields:**
- `repo`, `commit`, `centrality`, `structure`, `comp_id`, `comp_size`, `created_at`

**Only difference:** The node type (string module vs int goid).

**Recommendation:** Create a generic `SymbolMetricInputs[TNode]` using PEP 695 generics, similar to the orchestrator pattern.

**Impact:** 1 file
**Savings:** ~10 lines

---

#### 6.7.2 Extended Metric Inputs

| Dataclass | Location |
|-----------|----------|
| `FunctionMetricExtInputs` | `compute/row_builders/graph_metrics_ext.py:37-50` |
| `ModuleMetricExtInputs` | `compute/row_builders/graph_metrics_ext.py:53-64` |

**Shared fields:**
- `repo`, `commit`, `ctx`, `centralities`, `structure`, `components`

**Different fields:**
- Function: `articulations`, `bridge_incident`, `ancestor_count`, `descendant_count`
- Module: `rich_club`, `nodes`

**Recommendation:** Consider a base dataclass with shared fields + inheritance for specialization, or a Protocol-based approach.

**Impact:** 1 file
**Savings:** ~12 lines

---

### 6.8 CFG/DFG Context Consolidation 🟢

**Status:** 🟢 Lower Priority (Newly Identified)

**Issue:** CFG and DFG context dataclasses share many common fields:

| Context Class | Location | Lines |
|---------------|----------|-------|
| `CfgFnContext` | `cfg_dfg/cfg_core.py:35-50` | 16 |
| `DfgFnContext` | `cfg_dfg/dfg_core.py:38-64` | 27 |
| `CfgInputs` | `cfg_dfg/cfg_core.py:53-61` | 9 |
| `DfgInputs` | `cfg_dfg/dfg_core.py:66-76` | 11 |

**Common fields across all:**
- `repo`, `commit`, `now` (or `graph_ctx`)

**Common fields in FnContext:**
- `fn_goid`, `rel_path`, `module`, `qualname`, `graph`, `sccs`

**Recommendation:** Create a shared base `FnContext` in `cfg_dfg/types.py`:
```python
@dataclass(frozen=True)
class BaseFnContext:
    repo: str
    commit: str
    fn_goid: int
    rel_path: str
    module: str | None
    qualname: str | None
    graph: nx.DiGraph
    sccs: list[set[int]]
    now: datetime
```

**Impact:** 2 files + 1 new file
**Savings:** ~20 lines

---

### 6.9 Profile Writer Pattern Consolidation 🟡

**Status:** 🟡 Medium Priority (Newly Identified)

**Issue:** All profile writer functions follow the exact same pattern:

```python
def write_*_profile_rows(gateway, rows) -> int:
    rows_list = list(rows)
    if not rows_list:
        return 0
    repo = rows_list[0]["repo"]
    commit = rows_list[0]["commit"]
    context = WriterContext(
        table_key="analytics.*_profile",
        columns=*_PROFILE_COLUMNS,
        serialize_row=cast("SerializeRow", *_profile_row_to_tuple),
        repo=repo,
        commit=commit,
        ensure_schema_fn=lambda _gateway, _table: None,
    )
    return write_rows_with_registry_guard(gateway, rows=rows_list, context=context, ...)
```

**Instances:**
| Function | File |
|----------|------|
| `write_function_profile_rows` | `profiles/functions.py:975-1006` |
| `write_file_profile_rows` | `profiles/files.py:336-364` |
| `write_module_profile_rows` | `profiles/modules.py:372-402` |

**Recommendation:** Create a generic writer factory in `profiles/writer_guard.py`:
```python
def create_profile_writer(
    table_key: str,
    columns: Sequence[str],
    serialize_row: SerializeRow,
) -> Callable[[StorageGateway, Iterable[Mapping[str, object]]], int]:
    ...
```

**Impact:** 3 files + 1 utility function
**Savings:** ~40 lines

---

### 6.10 Remaining Persistence Migration 🟠

**Status:** 🟠 Overlooked in Phase 10 (Newly Identified)

**Issue:** `subsystems/materialize.py` still uses `gateway.ibis.write()` with manual column lists at lines 188-199:

```python
if membership_rows:
    gateway.ibis.write(
        "analytics.subsystem_modules",
        membership_rows,
        columns=SUBSYSTEM_MODULES_COLS,
    )

if subsystem_rows:
    gateway.ibis.write(
        "analytics.subsystems",
        subsystem_rows,
        columns=SUBSYSTEMS_COLS,
    )
```

**Recommendation:** Migrate to `DuckDBPolicyBackend.bulk_insert()` to match the standardized pattern.

**Impact:** 1 file
**Savings:** Column list maintenance + consistency

---

### 6.11 CFG/DFG Shared Helpers Module 🟡

**Status:** 🟡 Medium Priority (Newly Identified)

**Issue:** The `cfg_dfg/` package has several functions that are duplicated or closely related between `cfg_core.py` and `dfg_core.py`:

| Function | cfg_core.py | dfg_core.py | Action |
|----------|-------------|-------------|--------|
| `_degree_dict` | ✅ | ✅ | Consolidate |
| `parse_block_idx` | ✅ (public) | ✅ (private) | Consolidate + unify name |
| `function_metadata` | ✅ | ✅ (as `dfg_function_metadata`) | Consolidate |
| `loop_nodes` | ✅ | - | Keep in cfg |
| `loop_stats` | ✅ | - | Keep in cfg |
| `branching_stats` | ✅ | - | Keep in cfg |

**Recommendation:** Create `analytics/cfg_dfg/helpers.py`:
```python
"""Shared helpers for CFG and DFG analytics."""

def degree_dict(graph: nx.DiGraph, *, direction: str, weight: str | None = None) -> dict[int, int]:
    ...

def parse_block_idx(block_id: str | int | None) -> int | None:
    ...

def load_function_metadata(
    gateway: StorageGateway, repo: str, commit: str
) -> dict[int, tuple[str, str | None, str | None]]:
    ...
```

**Impact:** 2 files + 1 new file + 1 caller update
**Savings:** ~68 lines of duplicate code

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

### Completed Phases

#### Phase 9: Extended Constants Consolidation ✅

**Completed 2025-12-13**

- [x] Add `MAX_BETWEENNESS_NODES`, `MAX_COMMUNITY_NODES` to constants
- [x] Add `MAX_CFG_CENTRALITY_SAMPLE`, `MAX_DFG_CENTRALITY_SAMPLE`, `MAX_CFG_EIGEN_SAMPLE`
- [x] Update `symbol_graph_metrics.py` to import from constants
- [x] Update `config_graph_metrics.py` to import from constants
- [x] Update `cfg_dfg/materialize.py` to import from constants
- [x] Update `cfg_dfg/dfg_core.py` to import from constants

#### Phase 10: Persistence Pattern Standardization ✅

**Completed 2025-12-13**

Migrated 5 files from Pattern B (manual `gateway.ibis.write` with column lists) to use `DuckDBPolicyBackend.bulk_insert()`:

- [x] Migrate `subsystem_agreement.py` to use `bulk_insert()`
- [x] Migrate `subsystem_graph_metrics.py` to use `bulk_insert()`
- [x] Migrate `symbol_graph_metrics.py` to use `bulk_insert()`
- [x] Migrate `config_data_flow.py` to use `bulk_insert()`
- [x] Migrate `config_graph_metrics.py` to use `bulk_insert()`

**Key improvements:**
- Removed ~100 lines of manual column list definitions
- Standardized on `DuckDBPolicyBackend` for all persistence
- Schema-derived column order (no manual maintenance)
- Added numpy type conversion in `validate_tuple_rows()` for DuckDB compatibility

---

### Future Phases (Prioritized)

#### Phase 11: CFG/DFG Shared Helpers Module 🟡
**Priority: Medium | Savings: ~68 lines**
- [ ] Create `analytics/cfg_dfg/helpers.py`
- [ ] Move `_degree_dict` from both modules → `degree_dict` in helpers
- [ ] Move `parse_block_idx`/`_parse_block_idx` → unified `parse_block_idx` in helpers
- [ ] Consolidate `function_metadata`/`dfg_function_metadata` → `load_function_metadata`
- [ ] Update imports in `cfg_core.py`, `dfg_core.py`, `materialize.py`
- [ ] Run tests to verify no regressions

#### Phase 12: Duplicate Helper Function Consolidation 🔴
**Priority: Medium | Savings: ~18 lines**
- [ ] Create `analytics/utilities/dataframe.py` (or extend `profiles/utils.py`)
- [ ] Add `to_records(df: pd.DataFrame) -> list[dict[str, Any]]`
- [ ] Update `compute/row_builders/graph_metrics.py` to import from utilities
- [ ] Update `functions/metrics.py` to import from utilities
- [ ] Update `compute/functions/goids.py` to import from utilities
- [ ] Run tests to verify no regressions

#### Phase 13: Type Conversion Helper Unification 🟡
**Priority: Medium | Savings: ~15 lines**
- [ ] Create `analytics/utilities/type_coercion.py`
- [ ] Move `optional_str`, `optional_int`, `optional_float`, `optional_bool`, `int_or_default`
- [ ] Deprecate and redirect `_int_or_none` → `optional_int`
- [ ] Update `compute/row_builders/graph_metrics_ext.py`
- [ ] Re-export from `analytics/utilities/__init__.py`
- [ ] Update imports in `profiles/functions.py`, `profiles/files.py`, `profiles/modules.py`

#### Phase 14: Profile Writer Factory 🟡
**Priority: Medium | Savings: ~40 lines**
- [ ] Add `create_profile_writer()` factory to `profiles/writer_guard.py`
- [ ] Refactor `write_function_profile_rows` to use factory
- [ ] Refactor `write_file_profile_rows` to use factory
- [ ] Refactor `write_module_profile_rows` to use factory
- [ ] Run tests to verify no regressions

#### Phase 15: Remaining Persistence Migration 🟠
**Priority: High | Impact: 1 file**
- [ ] Migrate `subsystems/materialize.py` to use `DuckDBPolicyBackend.bulk_insert()`
- [ ] Remove `SUBSYSTEM_MODULES_COLS` and `SUBSYSTEMS_COLS` manual column lists
- [ ] Add dataset contracts to `utilities/datasets.py` if missing
- [ ] Run tests to verify no regressions

#### Phase 16: Row Builder Input Dataclass Unification 🟢
**Priority: Lower | Savings: ~22 lines**
- [ ] Create generic `SymbolMetricInputs[TNode]` using PEP 695
- [ ] Refactor `SymbolModuleMetricInputs` and `SymbolFunctionMetricInputs`
- [ ] Consider similar pattern for `FunctionMetricExtInputs`/`ModuleMetricExtInputs`

#### Phase 17: CFG/DFG Context Consolidation 🟢
**Priority: Lower | Savings: ~20 lines**
- [ ] Create `analytics/cfg_dfg/types.py` with `BaseFnContext`
- [ ] Refactor `CfgFnContext` to inherit from `BaseFnContext`
- [ ] Refactor `DfgFnContext` to inherit from `BaseFnContext`

#### Phase 18: Symbol Metrics Orchestrator 🟢
**Priority: Lower | Savings: ~50 lines**
- [ ] Design undirected metrics orchestrator abstraction
- [ ] Create `analytics/graphs/symbol_orchestrator.py`
- [ ] Refactor `compute_symbol_graph_metrics_modules` to use orchestrator
- [ ] Refactor `compute_symbol_graph_metrics_functions` to use orchestrator

---

## New Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `src/codeintel/analytics/graphs/constants.py` | Shared graph metrics constants | 46 |
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
| Duplicate constants | - | 2 sets consolidated ✅ |
| Lazy loading patterns | - | 3 → unified ✅ |
| Orchestration patterns | - | 2 → unified ✅ |
| Persistence patterns | - | Standardized via `bulk_insert()` ✅ (1 remaining) |
| **New utilities created** | - | 3 (constants.py, lazy_module.py, orchestrator.py) |
| **Duplicate helpers** | - | 7 identified → pending consolidation |
| **Type conversion** | - | 6 helpers → pending unification |
| **Profile writers** | - | 3 identical patterns → pending factory |
| **CFG/DFG helpers** | - | 4 duplicates → pending shared module |

---

## Summary of Identified Consolidation Opportunities

| Phase | Priority | Description | Savings |
|-------|----------|-------------|---------|
| 11 | 🟡 Medium | CFG/DFG shared helpers module | ~68 lines |
| 12 | 🔴 Medium | `_to_records` function consolidation | ~18 lines |
| 13 | 🟡 Medium | Type conversion helper unification | ~15 lines |
| 14 | 🟡 Medium | Profile writer factory pattern | ~40 lines |
| 15 | 🟠 High | Remaining persistence migration | Consistency |
| 16 | 🟢 Lower | Row builder dataclass unification | ~22 lines |
| 17 | 🟢 Lower | CFG/DFG context consolidation | ~20 lines |
| 18 | 🟢 Lower | Symbol metrics orchestrator | ~50 lines |
| **Total** | | | **~350+ lines** |

---

## Related Documents

- [GRAPHS_CLEANUP_PLAN.md](./GRAPHS_CLEANUP_PLAN.md)
- [BUILD_CLEANUP_PLAN.md](./BUILD_CLEANUP_PLAN.md)
- [BUILD_CONSOLIDATION_PLAN.md](./BUILD_CONSOLIDATION_PLAN.md)
- [BUILD_REFINEMENT_PLAN.md](./BUILD_REFINEMENT_PLAN.md)
