# Graphs Package Cleanup Plan

> **Generated:** 2025-12-13  
> **Updated:** 2025-12-13 (Phases 1-7c, 9a-b completed)  
> **Package:** `codeintel.graphs`  
> **Status:** Phases 1-9b Complete, Phase 8 + Future Work Pending

## Executive Summary

The `graphs` package has undergone extensive cleanup with Phases 1-7c and 9a-9b now complete:

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
- ~~Created `core/data_models/ids.py` with unified ID normalization functions~~ ✅
- ~~Created `core/ports` package with shared result base types~~ ✅
- ~~Migrated 10 test files to canonical types~~ ✅

**Remaining Opportunities (Phase 8+):**
- Remove deprecation shims after migration period (v6.0.0)
- Consolidate additional ID utilities (`to_decimal_id`, `normalize_node_id`)
- Consolidate function loading patterns
- Unify graph building utilities across packages

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

### Phase 7c: Test Migration ✅

**Completed 2025-12-13**

Migrated 10 test files from deprecated types to canonical types:

| File | Migration |
|------|-----------|
| `tests/analytics/integration/test_analytics_pipeline.py` | `FunctionCatalogService` → `CatalogService` |
| `tests/analytics/plugins/test_functions_effects_plugin.py` | `FunctionCatalogService` → `CatalogService` |
| `tests/analytics/plugins/test_dependencies_external_plugin.py` | `FunctionCatalogService` → `CatalogService` |
| `tests/analytics/plugins/test_entrypoints_plugin.py` | `FunctionCatalogService` → `CatalogService` |
| `tests/analytics/plugins/test_functions_plugins.py` | `FunctionCatalogService` → `CatalogService` |
| `tests/analytics/plugins/test_semantic_roles_plugin.py` | `FunctionCatalogService` → `CatalogService` |
| `tests/analytics/resources/test_provider_factory.py` | `FunctionMeta` → `FunctionSpan` |
| `tests/_helpers/rows.py` | `FunctionMeta` → `FunctionSpan` |
| `tests/_helpers/fakes/function_catalogs.py` | `FunctionMeta` → `FunctionSpan` |
| `tests/graphs/test_compute_layer.py` | `FunctionSpanData` → `FunctionSpan` |

### Phase 9a: ID Normalization Consolidation ✅

**Completed 2025-12-13**

Created unified ID normalization module at `core/data_models/ids.py`:

```
core/data_models/
├── __init__.py          # Exports normalize_decimal_id, as_int
├── ids.py               # Canonical ID normalization functions (NEW)
└── rows.py              # Row data models
```

**Changes:**
- Created `normalize_decimal_id()` - canonical function for DuckDB DECIMAL normalization
- Created `as_int()` - general integer coercion with bytes support
- Updated `graphs/engine/views.py` to import from `core.data_models.ids`
- Updated `analytics/compute/graphs/conversions.py` to re-export from core
- Maintains backward compatibility via aliases

### Phase 9b: Shared Ports Package ✅

**Completed 2025-12-13**

Created unified port protocols at `core/ports/`:

```
core/ports/
├── __init__.py          # Exports BaseQueryResult, BaseBatchResult
└── results.py           # Protocol definitions
```

**Protocols:**
- `BaseQueryResult` - protocol for query results with `row_count`
- `BaseBatchResult` - protocol for batch results with `rows_affected`

Both `graphs/ports/storage.py` and `ingestion/ports/storage.py` implementations now satisfy these protocols.

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
│   ├── CatalogService       # Unified service
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

core/
├── data_models/
│   ├── ids.py               # normalize_decimal_id, as_int (NEW)
│   └── rows.py              # Row data models
│
└── ports/
    ├── __init__.py          # BaseQueryResult, BaseBatchResult (NEW)
    └── results.py           # Protocol definitions (NEW)
```

### Canonical Types

| Canonical Type | Replaces | Location |
|----------------|----------|----------|
| `FunctionSpan` | `FunctionMeta`, `FunctionSpanData` | `graphs/catalog.py` |
| `CatalogService` | `FunctionCatalogService`, `CatalogResource` | `graphs/catalog.py` |
| `normalize_decimal_id` | `normalize_decimal` | `core/data_models/ids.py` |
| `BaseQueryResult` | (protocol) | `core/ports/results.py` |
| `BaseBatchResult` | (protocol) | `core/ports/results.py` |

---

## 3. Consumer Migration Status

### Completed Migrations ✅

All planned migrations are complete:

| Category | Files Migrated |
|----------|----------------|
| Analytics source files | 5 files |
| Test plugin files | 6 files |
| Test helper files | 4 files |
| **Total** | **15 files** |

### Remaining (Optional)

#### Build Plugins (Not Required)

| File | Current Usage | Notes |
|------|---------------|-------|
| `build/plugins/graphs/builders/callgraph.py` | `load_function_index` | Appropriate for span lookups |
| `build/plugins/graphs/builders/cfg_dfg.py` | `load_function_index` | Appropriate for span lookups |

**Decision:** Keep as-is. The build plugins use `load_function_index` which returns `FunctionSpanIndex` - the appropriate type for span lookups.

---

## 4. Cross-Package Consolidation

### 4.1 Completed: ID Normalization ✅

**Status:** Consolidated to `core/data_models/ids.py`

| Function | Location | Purpose |
|----------|----------|---------|
| `normalize_decimal_id()` | `core/data_models/ids.py` | DuckDB DECIMAL → int |
| `as_int()` | `core/data_models/ids.py` | General int coercion |

**Consumers:**
- `graphs/engine/views.py` - imports as `normalize_decimal`
- `analytics/compute/graphs/conversions.py` - re-exports for backward compatibility
- `analytics/cfg_dfg/helpers.py` - uses via `analytics/compute/graphs`

### 4.2 Completed: Shared Port Protocols ✅

**Status:** Created `core/ports` package

| Protocol | Location | Purpose |
|----------|----------|---------|
| `BaseQueryResult` | `core/ports/results.py` | Query result with `row_count` |
| `BaseBatchResult` | `core/ports/results.py` | Batch result with `rows_affected` |

**Implementations:**
- `graphs/ports/storage.py` - `QueryResult`, `BatchResult`
- `ingestion/ports/storage.py` - `QueryResult`, `BatchResult`

### 4.3 Future: Additional ID Utilities

**Finding:** Two related functions exist in `analytics/compute/graphs/conversions.py` that could move to `core/data_models/ids.py`:

| Function | Purpose | Effort |
|----------|---------|--------|
| `to_decimal_id()` | Convert int/str to `Decimal` for DuckDB writes | Low |
| `normalize_node_id()` | Normalize graph node IDs to int/str | Low |

```python
# Candidate for core/data_models/ids.py
def to_decimal_id(value: int | str | Decimal | None) -> Decimal | None:
    """Coerce identifiers to Decimal for DuckDB writes."""
    if value is None:
        return None
    return Decimal(int(value))

def normalize_node_id(node: Decimal | float | str | None) -> int | str | None:
    """Normalize graph node identifiers for consistent dictionary keys."""
    ...
```

**Benefit:** Complete ID utility consolidation in one module
**Effort:** 30 minutes

### 4.4 Future: Function Loading Patterns

**Finding:** Multiple modules implement function loading from the database:

| Location | Function | Returns |
|----------|----------|---------|
| `graphs/catalog.py` | `load_function_spans()` | `list[FunctionSpan]` |
| `graphs/catalog.py` | `load_function_index()` | `FunctionSpanIndex` |
| `graphs/catalog.py` | `load_function_catalog()` | `FunctionCatalog` |
| `analytics/cfg_dfg/helpers.py` | `load_function_metadata()` | `dict[int, tuple]` |
| `analytics/parsing/ast_cache.py` | `load_function_asts()` | `dict[int, FunctionAst]` |
| `analytics/profiles/functions.py` | `load_function_base_info()` | Profile info |

**Observation:** These serve different purposes but share common patterns:
- All query `core.goids` with repo/commit filters
- All normalize GOID columns
- All build dicts or lists keyed by GOID

**Potential consolidation:**
```python
# graphs/catalog.py - add thin helper
def load_function_rows(gateway, *, repo, commit, columns="*") -> Iterator[tuple]:
    """Base row iterator for function queries."""
    ...

# Other modules can use this as foundation
```

**Effort:** Medium (2-3 hours)
**Benefit:** Reduced SQL duplication, consistent normalization

### 4.5 Future: Graph Building Utilities

**Finding:** 24 graph-building functions across packages:

```
graphs/engine/views.py           - 7 graph creation sites (nx.DiGraph/Graph)
analytics/compute/graphs/cfg.py  - build_cfg_graph()
analytics/compute/graphs/dfg.py  - build_dfg_graph()
analytics/subsystems/affinity.py - build_weighted_graph()
+ 14 more build_*_graph functions
```

**Common patterns:**
1. Create empty graph: `nx.DiGraph()` or `nx.Graph()`
2. Add nodes with attributes
3. Add edges with weights
4. Return graph

**Potential utility:**
```python
# core/graph_utils.py (or similar)
def build_directed_graph(
    nodes: Iterable[tuple[int, dict]],
    edges: Iterable[tuple[int, int, dict]],
) -> nx.DiGraph:
    """Construct a directed graph from nodes and edges."""
    ...
```

**Effort:** High (4-6 hours)
**Benefit:** Consistent graph construction patterns

### 4.6 Future: Type Coercion Utilities

**Finding:** Multiple `safe_*` functions scattered across packages:

| Function | Location | Purpose |
|----------|----------|---------|
| `safe_float()` | `analytics/compute/graphs/conversions.py` | Float coercion |
| `safe_relpath()` | `ingestion/infrastructure/paths.py` | Safe relative path |
| `safe_ratio()` | `analytics/compute/ibis_utils.py` | Safe division |
| `safe_count()` | `ingestion/infrastructure/db_queries.py` | Safe DB count |
| `safe_unparse()` | `analytics/utilities/ast.py` | Safe AST unparse |
| + 10 more | `ingestion/infrastructure/db_queries.py` | Safe DB operations |

**Observation:** These follow a consistent pattern:
```python
def safe_X(value: T | None) -> R | None:
    """Safe coercion that never raises."""
    if value is None:
        return None
    try:
        return convert(value)
    except Exception:
        return None
```

**Potential consolidation:**
```python
# core/utils/safe.py
def safe_coerce[T, R](value: T | None, converter: Callable[[T], R]) -> R | None:
    """Generic safe coercion wrapper."""
    ...
```

**Effort:** Medium (2-3 hours)
**Benefit:** Consistent error handling, reduced boilerplate

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

### Removal Checklist (Phase 8)

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

**Current state:** Well-structured with clear separation:
- `runner.py` - Orchestration
- `checks.py` - Individual validators
- `findings.py` - Finding types and persistence

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

### 6.5 Resource Provider Pattern Standardization

**Finding:** All resource providers follow the same pattern:

```python
class XProvider(LazyResource[T]):
    RESOURCE_NAME: ClassVar[str] = "name"
    
    def _load(self) -> T: ...
    def get(self) -> T: ...
```

**Current implementations:**
- `CatalogProvider` → `FunctionCatalog`
- `GraphProvider` → `GraphResources`
- `AstProvider` → `AstResourceData`
- `FeaturesProvider` → `dict[int, FunctionAstFeatures]`
- `ModuleMapProvider` → `dict[str, str]`

**Observation:** Pattern is already clean and consistent. No consolidation needed.

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
- [x] Update `compute/__init__.py` docstring

#### Phase 7a: Core Consumer Migration ✅
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

#### Phase 7c: Test Migration ✅
- [x] Migrate `tests/analytics/plugins/test_*.py` (6 files)
- [x] Migrate `tests/analytics/resources/test_provider_factory.py`
- [x] Migrate `tests/_helpers/rows.py`
- [x] Migrate `tests/_helpers/fakes/function_catalogs.py`
- [x] Migrate `tests/graphs/test_compute_layer.py`
- [x] Run full test suite (719 graph tests pass)

#### Phase 9a: ID Normalization Consolidation ✅
- [x] Create `core/data_models/ids.py` with `normalize_decimal_id` and `as_int`
- [x] Update `graphs/engine/views.py` to import from `core.data_models.ids`
- [x] Update `analytics/compute/graphs/conversions.py` to re-export from core
- [x] Update `core/data_models/__init__.py` exports

#### Phase 9b: Shared Ports Package ✅
- [x] Create `core/ports/__init__.py` with `BaseQueryResult`, `BaseBatchResult`
- [x] Create `core/ports/results.py` with protocol definitions
- [x] Update `graphs/ports/storage.py` with protocol documentation
- [x] Update `ingestion/ports/storage.py` with protocol documentation

### Remaining Phases

#### Phase 8: Deprecation Removal (After v6.0.0)
- [ ] Remove `FunctionMeta` compatibility wrapper
- [ ] Remove `FunctionCatalogService` compatibility wrapper
- [ ] Remove `FunctionSpanData` compatibility wrapper
- [ ] Remove `CatalogResource` wrapper
- [ ] Remove deprecated port protocols
- [ ] Update all `__all__` exports

#### Phase 10: Additional ID Utility Consolidation (Optional)
- [ ] Move `to_decimal_id()` to `core/data_models/ids.py`
- [ ] Move `normalize_node_id()` to `core/data_models/ids.py`
- [ ] Move `safe_float()` to `core/data_models/ids.py` or `core/utils/coerce.py`
- [ ] Update imports in `analytics/compute/graphs/conversions.py`
- [ ] Update imports in `analytics/compute/row_builders/*.py`

#### Phase 11: Function Loading Consolidation (Optional)
- [ ] Create `load_function_rows()` base helper in `graphs/catalog.py`
- [ ] Refactor `load_function_metadata()` in `analytics/cfg_dfg/helpers.py`
- [ ] Assess other function loading patterns

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

After Phase 9b completion:
- ✅ `uv run pyright` - 0 errors on all modified files
- ✅ `uv run ruff check --fix` - All checks passed
- ✅ `uv run pytest tests/graphs/` - 719 passed
- ✅ `uv run pytest tests/analytics/plugins/` - 9 passed
- ✅ `uv run pytest tests/analytics/resources/test_provider_factory.py` - 30 passed

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
