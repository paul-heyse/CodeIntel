# Graphs Package Cleanup Plan

> **Generated:** 2025-12-13  
> **Updated:** 2025-12-13 (Phases 1-7a completed, test fixtures fixed)  
> **Package:** `codeintel.graphs`  
> **Status:** Phases 1-7a Complete, Phase 7b-9 Pending

## Executive Summary

The `graphs` package has undergone extensive cleanup with Phases 1-7a now complete:

**Completed:**
- ~~2 empty directories deleted~~ ✅
- ~~9 unused backward-compatibility aliases removed~~ ✅
- ~~1 deprecated stub module removed~~ ✅
- ~~1 unused protocol removed~~ ✅
- ~~3 data classes unified into FunctionSpan~~ ✅
- ~~2 service classes merged into CatalogService~~ ✅
- ~~3 port protocols marked deprecated~~ ✅
- ~~compute/__init__.py docstring updated~~ ✅
- ~~Fixed pyright error in resolution.py (FunctionSpanData → FunctionSpan)~~ ✅
- ~~Migrated 5 analytics files to use CatalogService~~ ✅
- ~~Fixed missing `graph_executor_env` fixture and `GraphTestEnv` type~~ ✅

**Remaining Opportunities (Phase 7b+):**
- Migrate build plugins from load_function_index to load_function_catalog
- Migrate test files to use CatalogService
- Remove deprecation shims after migration period (v6.0.0)
- Consolidate duplicate normalize_decimal functions
- Cross-package port unification (analytics/ingestion)
- Test helper modernization

---

## Table of Contents

1. [Completed Work](#1-completed-work)
2. [Current Architecture](#2-current-architecture)
3. [Consumer Migration Status](#3-consumer-migration-status)
4. [Cross-Package Consolidation](#4-cross-package-consolidation)
5. [Deprecation Removal Timeline](#5-deprecation-removal-timeline)
6. [Future Opportunities](#6-future-opportunities)
7. [Implementation Checklist](#7-implementation-checklist)

---

## 1. Completed Work

### Phase 1: Dead Directories and Aliases ✅

**Completed 2025-12-13**

- Deleted empty directories: `core/`, `runtime/`
- Removed 9 unused backward-compatibility aliases:
  - 4 from `graphs/engine/views.py`
  - 4 from `graphs/compute/callgraph/resolution.py`
  - 1 from `graphs/compute/callgraph/collection.py`

### Phase 2: Deprecated Adapters Removal ✅

**Completed 2025-12-13**

- Deleted `src/codeintel/graphs/adapters/` directory
- Updated `graphs/__init__.py` to remove adapters import and export

### Phase 3: ParsingPort Protocol Removal ✅

**Completed 2025-12-13**

- Removed unused `ParsingPort` protocol from `graphs/ports/parsing.py`
- Kept data classes: `ParsedFunction`, `ParsedModule`, `ParseError`, `ParseResult`

### Phase 4: Catalog Layer Consolidation ✅

**Completed 2025-12-13**

- **Unified FunctionSpan**: Added optional `urn` field and `local_name` property
- **Updated FunctionCatalog**: Now accepts `Iterable[FunctionSpan]` directly
- **Created CatalogService**: Merged `FunctionCatalogService` and `CatalogResource`
- **Added deprecation wrappers**:
  - `FunctionMeta` → compatibility class returning `FunctionSpan`
  - `FunctionCatalogService` → wrapper returning `CatalogService`
  - `FunctionSpanData` → compatibility class in `ports/catalog.py`

### Phase 5: Ports Layer Simplification ✅

**Completed 2025-12-13**

- Marked protocols as deprecated with migration guidance:
  - `StoragePort` in `ports/storage.py`
  - `CatalogPort` in `ports/catalog.py`
  - `EnginePort` in `ports/engine.py`
- Kept active data classes: `QueryResult`, `BatchResult`, `GraphData`
- Updated `ports/__init__.py` with sorted exports and deprecation notes

### Phase 6: Polish ✅

**Completed 2025-12-13**

- Updated `compute/__init__.py` docstring to reflect current structure
- Added subpackage documentation for `callgraph/` and `metrics/`

### Phase 7a-ext: Test Fixture Recovery ✅

**Completed 2025-12-13**

During Phase 2 of the Legacy Decommissioning Plan (Hamilton integration), the `tests/_helpers/fakes/graph_contexts.py` file was deleted but tests still referenced the `graph_executor_env` fixture. This fix:

- Created `tests/_helpers/fakes/graph_contexts.py` with `GraphTestEnv` dataclass
- Added `graph_executor_env` fixture to `tests/graphs/conftest.py`
- Fixed 26 graph tests that were failing due to missing fixture

### Phase 7a-ext2: Legacy API Migration & Bug Fixes ✅

**Completed 2025-12-13**

Fixed `test_span_consistency_integration.py` which was using deprecated `ConfigBuilder.analytics.test_coverage()` API:

- Updated test to use `TestCoverageOptions` directly with `compute_test_coverage_edges()`
- Added `TestCoverageOptions` to `codeintel.analytics.testing.__init__.py` exports
- Fixed schema mismatch bug in `TEST_CATALOG_UPDATE_GOIDS` SQL statement:
  - Changed `function_goid_h128` → `test_goid_h128` to match `test_catalog` schema
- All 719 graph tests now pass

---

## 2. Current Architecture

### Post-Consolidation Structure

```
graphs/
├── __init__.py              # Exports CatalogService, FunctionSpan
├── catalog.py               # Unified catalog layer
│   ├── FunctionSpan         # Unified data class with urn + local_name
│   ├── FunctionSpanIndex    # Lookup structure
│   ├── FunctionCatalog      # Main catalog class
│   ├── FunctionCatalogProvider  # Protocol for DI
│   ├── CatalogService       # Unified service (NEW)
│   ├── FunctionMeta         # DEPRECATED compatibility wrapper
│   └── FunctionCatalogService   # DEPRECATED compatibility wrapper
│
├── ports/
│   ├── __init__.py          # Re-exports data classes
│   ├── catalog.py           # CatalogPort (deprecated), FunctionSpanData (deprecated)
│   ├── engine.py            # EnginePort (deprecated), GraphData (active)
│   ├── parsing.py           # ParsedFunction, ParsedModule (active)
│   └── storage.py           # StoragePort (deprecated), QueryResult, BatchResult (active)
│
├── resources/
│   ├── __init__.py          # Exports CatalogService, resources
│   ├── catalog.py           # CatalogResource (deprecated shim → CatalogService)
│   ├── graphs.py            # GraphResource (active)
│   └── storage.py           # StorageResource (active)
│
├── compute/                 # Pure stateless computation layer
│   ├── callgraph/           # Edge collection, resolution, deduplication
│   ├── metrics/             # Graph metric computations
│   ├── cfg.py, dfg.py       # Control/data flow graph construction
│   ├── goid.py              # GOID hash computation
│   ├── imports.py           # Import analysis
│   └── symbols.py           # Symbol use analysis
│
├── engine/                  # Graph engine implementations
└── validation/              # Graph validation checks
```

### New Canonical Types

| Canonical Type | Replaces | Location |
|----------------|----------|----------|
| `FunctionSpan` | `FunctionMeta`, `FunctionSpanData` | `graphs/catalog.py` |
| `CatalogService` | `FunctionCatalogService`, `CatalogResource` | `graphs/catalog.py` |

---

## 3. Consumer Migration Status

### Completed Migrations ✅

The following files have been migrated from `FunctionCatalogService` to `CatalogService`:

| File | Status |
|------|--------|
| `analytics/resources/catalog.py` | ✅ Migrated |
| `analytics/testing/coverage/edges.py` | ✅ Migrated |
| `analytics/profiles/__init__.py` | ✅ Migrated |
| `analytics/functions/function_effects.py` | ✅ Migrated |
| `analytics/parsing/ast_cache.py` | ✅ Migrated |
| `graphs/compute/callgraph/resolution.py` | ✅ Fixed (FunctionSpanData → FunctionSpan) |

### Remaining Migrations

#### Medium Priority (Build Plugins)

| File | Deprecated Import | Migration |
|------|-------------------|-----------|
| `build/plugins/graphs/builders/callgraph.py` | `load_function_index` | Use `load_function_catalog` |
| `build/plugins/graphs/builders/cfg_dfg.py` | `load_function_index` | Use `load_function_catalog` |

**Note:** The build plugins use `load_function_index` which returns `FunctionSpanIndex`. This is acceptable as-is since the index is needed for span lookups. Consider consolidating after `load_function_catalog` can efficiently serve both use cases.

#### Lower Priority (Tests)

| File | Deprecated Import | Migration |
|------|-------------------|-----------|
| `tests/analytics/plugins/test_*.py` (6 files) | `FunctionCatalogService` | Use `CatalogService` |
| `tests/analytics/resources/test_provider_factory.py` | `FunctionMeta` | Use `FunctionSpan` |
| `tests/graphs/conftest.py` | `FunctionMeta` | Use `FunctionSpan` |
| `tests/_helpers/rows.py` | `FunctionMeta` | Use `FunctionSpan` |
| `tests/_helpers/fakes/function_catalogs.py` | `FunctionMeta` | Use `FunctionSpan` |

---

## 4. Cross-Package Consolidation

### 4.1 Redundant Port Re-Exports

During implementation, I discovered that multiple packages maintain their own port modules that simply re-export from `graphs.ports`:

```
analytics/ports/__init__.py  → re-exports from graphs.ports (CORRECT pattern)
ingestion/ports/storage.py   → has DIFFERENT BatchResult/QueryResult definitions
```

**Note:** The `analytics.ports` pattern is correct and should be preserved. The `ingestion.ports` definitions are different and serve a different purpose (ingestion-specific batch operations).

### 4.2 Duplicate normalize_decimal Functions

**New Finding (2025-12-13):** Two nearly identical functions exist for converting DuckDB DECIMAL values:

| Location | Function | Purpose |
|----------|----------|---------|
| `graphs/engine/views.py` | `normalize_decimal()` | Graph loading from DuckDB |
| `analytics/compute/graphs/conversions.py` | `normalize_decimal_id()` | Analytics graph computations |

**Code comparison:**
```python
# Both functions are semantically identical:
# - Accept object value
# - Handle None, int, Decimal, bytes, str
# - Return int | None
```

**Recommendation:** Consolidate into a shared utility module:

```python
# Option A: codeintel/storage/helpers/decimal.py (storage layer)
# Option B: codeintel/core/utils/decimal.py (core utilities)

def normalize_decimal(value: object) -> int | None:
    """Normalize DuckDB DECIMAL(38,0) values to Python ints."""
    ...
```

Then have both `graphs/engine/views.py` and `analytics/compute/graphs/conversions.py` import from there.

**Effort:** Low (30 minutes)
**Benefit:** Single source of truth, reduced maintenance

### 4.3 Port Unification Recommendation

Consider a single shared ports package for common data types only:

```python
# codeintel/core/ports/__init__.py (or similar)
from codeintel.graphs.ports import (
    FunctionSpan,
    GraphData,
    ParsedFunction,
    ParsedModule,
)
```

**Benefits:**
- Single source of truth for shared data classes
- Clearer import paths
- Reduced maintenance burden

**Effort:** Low (1-2 hours)

---

## 5. Deprecation Removal Timeline

### Current Deprecation Status

All deprecated items emit `DeprecationWarning` at runtime:

| Deprecated Item | Location | Replacement | Remove After |
|-----------------|----------|-------------|--------------|
| `FunctionMeta` | `graphs/catalog.py` | `FunctionSpan` | v6.0.0 |
| `FunctionCatalogService` | `graphs/catalog.py` | `CatalogService` | v6.0.0 |
| `FunctionSpanData` | `graphs/ports/catalog.py` | `FunctionSpan` | v6.0.0 |
| `CatalogResource` | `graphs/resources/catalog.py` | `CatalogService` | v6.0.0 |
| `CatalogPort` | `graphs/ports/catalog.py` | `CatalogService` | v6.0.0 |
| `StoragePort` | `graphs/ports/storage.py` | `StorageResource` | v6.0.0 |
| `EnginePort` | `graphs/ports/engine.py` | `GraphResource` | v6.0.0 |

### Removal Checklist (Phase 7)

After all consumers migrate:

1. Remove `_FunctionMetaCompat` class and `FunctionMeta` alias from `catalog.py`
2. Remove `FunctionCatalogService` wrapper from `catalog.py`
3. Remove `_FunctionSpanDataCompat` class and `FunctionSpanData` alias from `ports/catalog.py`
4. Remove `CatalogResource` wrapper from `resources/catalog.py`
5. Remove `CatalogPort`, `StoragePort`, `EnginePort` protocols
6. Update all `__all__` exports

---

## 6. Future Opportunities

### 6.1 Loading Function Consolidation

**Current state:**
```python
load_function_spans()   # Returns list[FunctionSpan] without URN
load_function_index()   # Returns FunctionSpanIndex
load_function_catalog() # Returns FunctionCatalog with URN
```

**Observation:** `load_function_spans()` is now redundant since `FunctionSpan` includes URN.

**Recommendation:** After migration period, consolidate to:
```python
load_function_catalog()  # Primary loader
# load_function_index() and load_function_spans() can delegate to it
```

### 6.2 Test Helper Modernization

**Observation:** `tests/_helpers/catalogs.py` was updated during implementation but still uses:
- Direct `FunctionCatalog` iteration patterns
- Manual row building that duplicates catalog logic

**Recommendation:** Simplify test helpers to use `CatalogService` directly:
```python
def seed_goids_from_catalog(ctx: CatalogCtxLike, catalog: CatalogService) -> None:
    # Use catalog.function_spans directly instead of internal iteration
    for span in catalog.function_spans:
        ...
```

### 6.3 Validation Package Assessment

**Observation:** The `graphs/validation/` package was not touched in this cleanup.

**Potential opportunities:**
- Check if validation uses deprecated types
- Assess if validation can use `CatalogService` directly
- Look for dead code or unused validators

### 6.4 Engine Package Assessment

**Observation:** The `graphs/engine/` package was not deeply analyzed.

**Potential opportunities:**
- Check for unused backward-compatibility code
- Assess if `GraphEngine` protocol can be simplified
- Look for opportunities to use `CatalogService`

---

## 7. Implementation Checklist

### Completed Phases ✅

#### Phase 1: Dead Directories and Aliases ✅
- [x] Delete `src/codeintel/graphs/core/` directory
- [x] Delete `src/codeintel/graphs/runtime/` directory
- [x] Remove 4 aliases from `graphs/engine/views.py`
- [x] Remove 4 aliases from `graphs/compute/callgraph/resolution.py`
- [x] Remove 1 alias from `graphs/compute/callgraph/collection.py`

#### Phase 2: Deprecated Module Removal ✅
- [x] Remove `src/codeintel/graphs/adapters/` directory
- [x] Update `src/codeintel/graphs/__init__.py` to remove adapters

#### Phase 3: Protocol Simplification ✅
- [x] Remove `ParsingPort` protocol from `graphs/ports/parsing.py`
- [x] Update `graphs/ports/__init__.py` exports

#### Phase 4: Catalog Layer Consolidation ✅
- [x] Add `urn` field and `local_name` property to `FunctionSpan`
- [x] Update `FunctionCatalog` to use unified `FunctionSpan`
- [x] Create unified `CatalogService` class
- [x] Add deprecation wrappers for `FunctionMeta`, `FunctionCatalogService`, `FunctionSpanData`
- [x] Update `resources/catalog.py` to be deprecation shim
- [x] Update `resources/__init__.py` exports
- [x] Update `graphs/__init__.py` exports
- [x] Update `analytics/ports/__init__.py` exports
- [x] Update `tests/_helpers/catalogs.py`

#### Phase 5: Ports Layer Simplification ✅
- [x] Mark `StoragePort` as deprecated in `ports/storage.py`
- [x] Mark `CatalogPort` as deprecated in `ports/catalog.py`
- [x] Mark `EnginePort` as deprecated in `ports/engine.py`
- [x] Update `ports/__init__.py` with deprecation notes

#### Phase 6: Polish ✅

**Completed 2025-12-13**

- [x] Update `compute/__init__.py` docstring

#### Phase 7a: Core Consumer Migration ✅

**Completed 2025-12-13**

- [x] Fix pyright error in `graphs/compute/callgraph/resolution.py` (FunctionSpanData → FunctionSpan)
- [x] Migrate `analytics/resources/catalog.py` to use `CatalogService`
- [x] Migrate `analytics/testing/coverage/edges.py`
- [x] Migrate `analytics/profiles/__init__.py`
- [x] Migrate `analytics/functions/function_effects.py`
- [x] Migrate `analytics/parsing/ast_cache.py`
- [x] Run pyright and ruff (all checks pass)
- [x] Fix missing `graph_executor_env` fixture (create `tests/_helpers/fakes/graph_contexts.py`)
- [x] Add `GraphTestEnv` dataclass for graph integration tests
- [x] Update `tests/graphs/conftest.py` with new fixture
- [x] Update `test_span_consistency_integration.py` to use new API
- [x] Export `TestCoverageOptions` from `analytics.testing`
- [x] Fix `TEST_CATALOG_UPDATE_GOIDS` schema mismatch bug

### Remaining Phases

#### Phase 7b: Build Plugin Migration (Optional)
- [ ] Assess whether `load_function_index` should migrate to `load_function_catalog`
- [ ] If yes, migrate `build/plugins/graphs/builders/callgraph.py`
- [ ] If yes, migrate `build/plugins/graphs/builders/cfg_dfg.py`

**Note:** Build plugins use `load_function_index` for span lookups. This is acceptable since `FunctionSpanIndex` is the appropriate type for those use cases.

#### Phase 7c: Test Migration
- [ ] Migrate `tests/analytics/plugins/test_*.py` (6 files)
- [ ] Migrate `tests/analytics/resources/test_provider_factory.py`
- [ ] Migrate `tests/graphs/conftest.py`
- [ ] Migrate `tests/_helpers/rows.py`
- [ ] Migrate `tests/_helpers/fakes/function_catalogs.py`
- [ ] Run full test suite

#### Phase 8: Deprecation Removal (After v6.0.0)
- [ ] Remove `FunctionMeta` compatibility wrapper
- [ ] Remove `FunctionCatalogService` compatibility wrapper
- [ ] Remove `FunctionSpanData` compatibility wrapper
- [ ] Remove `CatalogResource` wrapper
- [ ] Remove deprecated port protocols
- [ ] Update all `__all__` exports

#### Phase 9: Cross-Package Consolidation (Optional)
- [ ] Consolidate duplicate `normalize_decimal` functions (see section 4.2)
- [ ] Assess shared `core/ports` package for common data types
- [ ] Assess `ingestion/ports` patterns

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
uv run vulture src/codeintel/graphs --min-confidence 90

# Check for remaining deprecated usage in src/
grep -r "FunctionCatalogService\|FunctionMeta\|FunctionSpanData" src/ --include="*.py" | grep -v "# Deprecated"
```

### Verification Status (2025-12-13)

After Phase 7a completion:
- ✅ `uv run pyright --warnings --pythonversion=3.13 src/codeintel/graphs src/codeintel/analytics` - 0 errors
- ✅ `uv run ruff check --fix src/codeintel/graphs src/codeintel/analytics` - All checks passed

Remaining deprecated usage in `src/`:
- `analytics/ports/__init__.py` - Re-export layer (intentional)
- `analytics/functions/metrics.py` - Has its own `FunctionMeta` class (different purpose)
- `graphs/catalog.py`, `graphs/ports/catalog.py` - Deprecation wrappers (intentional)

---

## Related Documents

- [BUILD_CLEANUP_PLAN.md](./BUILD_CLEANUP_PLAN.md)
- [BUILD_CONSOLIDATION_PLAN.md](./BUILD_CONSOLIDATION_PLAN.md)
- [BUILD_REFINEMENT_PLAN.md](./BUILD_REFINEMENT_PLAN.md)
- [ANALYTICS_CLEANUP_PLAN.md](./ANALYTICS_CLEANUP_PLAN.md)
- [INGESTION_CLEANUP_PLAN.md](./INGESTION_CLEANUP_PLAN.md)
