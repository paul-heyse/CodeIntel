# Hamilton Consolidation — Remaining Scope for Legacy Deletion

## Executive Summary

Phase 3 of the Hamilton Consolidation (PR-66 through PR-73) is complete. The core infrastructure is in place:
- ✅ Schema provider with unified fallback chain
- ✅ Generated row bindings (not hand-maintained)
- ✅ Target-derived contracts
- ✅ Schema drift CI gate
- ✅ Unified catalog (tables, views, artifacts)
- ✅ JSON Schema generation from TableSchema
- ✅ Export code consolidated to `build/exports/`

However, several legacy files in `config/datasets/` remain because **TypedDict row models** are still imported throughout the codebase. This document details the remaining scope to fully delete the legacy infrastructure.

---

## Current State of `config/datasets/`

```
config/datasets/
├── __init__.py           # Re-exports, can simplify after migration
├── columns.py            # Column definitions - assess if still needed
├── composites.py         # COMPOSITE_SCHEMAS - moved from schemas.py, may need to relocate
├── contracts.py          # Re-exports DatasetContract, RowBinding - keep as shim or delete
├── dataflow.py           # Lazy import helpers - assess if still needed
├── dependencies.py       # Dependency tracking - assess if still needed
├── primitives.py         # Re-exports from core/schemas - keep as shim or delete
├── schemas.py            # TABLE_SCHEMAS (~114KB) - BLOCKING DELETION
├── semantic_roles.py     # Semantic role definitions - assess if still needed
├── rows/                 # Hand-maintained TypedDicts - BLOCKING DELETION
│   ├── __init__.py
│   ├── analytics.py      # FunctionMetricsRow, CoverageLineRow, etc.
│   ├── core.py           # GoidRow, ModuleRow, etc.
│   ├── graph.py          # CallGraphEdgeRow, ImportEdgeRow, etc.
│   ├── profiles.py       # GraphMetricsFunctionsRow, GraphMetricsModulesRow
│   └── test.py           # BehavioralCoverageRowModel, TestCoverageEdgeRow
└── generated_rows/       # Schema-prefixed auto-generated TypedDicts
    ├── __init__.py
    ├── analytics.py      # AnalyticsFunctionMetricsRow, etc.
    ├── core.py           # CoreGoidRow, etc.
    └── graph.py          # GraphCallGraphEdgesRow, etc.
```

---

## Blocking Dependencies

### 1. `schemas.py` (TABLE_SCHEMAS)

**Why it still exists**: `rows/analytics.py` imports `TABLE_SCHEMAS` via lazy import in `_get_contract_columns()`:

```python
def _get_contract_columns(table_key: str) -> tuple[str, ...]:
    from codeintel.config.datasets.schemas import TABLE_SCHEMAS  # noqa: PLC0415
    schema = TABLE_SCHEMAS.get(table_key)
    if schema is None:
        message = f"No schema defined for table key: {table_key}"
        raise ValueError(message)
    return tuple(schema.column_names())
```

**Consumers**: ~12 column constant definitions in `rows/analytics.py` use this function to derive column tuples at module load time.

**Size**: ~114KB, ~3000+ lines

### 2. `rows/` directory (TypedDict definitions)

**Why it still exists**: Many files import TypedDict row models directly:

| File | Imports |
|------|---------|
| `analytics/compute/hotspots/metrics.py` | `FunctionMetricsRow` |
| `analytics/compute/row_builders/*.py` | Various row types |
| `graphs/compute/callgraph/collection.py` | `CallGraphEdgeRow` |
| `graphs/compute/callgraph/persistence.py` | `CallGraphEdgeRow` |
| `build/plugins/graphs/builders/callgraph.py` | `CallGraphEdgeRow` |
| `ingestion/compute/typing_ingest.py` | `TypeHintRow` |
| `tests/_helpers/analytics_domain.py` | `CoverageLineRow` |
| `tests/_helpers/factories/row_factories.py` | Various row types |
| `tests/_helpers/seeds/architecture.py` | `GraphMetricsFunctionsRow` |
| `tests/docs_export/*.py` | Various row types (TYPE_CHECKING imports) |
| `tests/graphs/*.py` | `CallGraphEdgeRow` (TYPE_CHECKING imports) |

**Size**: ~50KB across 5 files

### 3. `generated_rows/` directory

**Why it still exists**: Contains auto-generated schema-prefixed TypedDicts. These were created as a migration target but are rarely used. Most code still imports from `rows/`.

**Size**: ~30KB across 3 files

### 4. `contracts.py`

**Why it still exists**: Re-exports `DatasetContract` and `RowBinding` from `core/schemas/contract_primitives.py`. Some legacy imports may still reference this location.

**Size**: ~200 lines (mostly re-exports and metadata dictionaries)

---

## Migration Strategy

### Option A: Generate TypedDicts at Runtime (Recommended)

**Approach**: Extend `core/schemas/row_models.py` to generate TypedDict classes dynamically:

```python
from codeintel.build.schemas import get_schema_provider

def get_row_type(table_key: str) -> type[TypedDict]:
    """Return a dynamically generated TypedDict for a table key."""
    schema = get_schema_provider().require_table_schema(table_key)
    return _generate_typed_dict_for_schema(schema)
```

**Pros**:
- Single source of truth (TableSchema)
- No hand-maintained TypedDict definitions
- Automatic sync with schema changes

**Cons**:
- Runtime type generation has limitations (IDE autocomplete, static analysis)
- May need to pre-generate and cache for performance
- TYPE_CHECKING imports won't work with runtime generation

### Option B: Static Generation with Build-Time Codegen

**Approach**: Add a build step that generates `.py` files from TableSchema:

```bash
codeintel build generate-row-models --output src/codeintel/core/schemas/generated_row_types.py
```

**Pros**:
- Full IDE support and static analysis
- Single source of truth (TableSchema → generated code)
- Clear separation of generated vs. hand-maintained code

**Cons**:
- Need to regenerate on schema changes
- Risk of generated code drifting from schemas
- More complex build process

### Option C: Gradual Migration with Compatibility Layer

**Approach**: Keep `rows/` as compatibility layer, but generate content from TableSchema:

```python
# rows/analytics.py
from codeintel.core.schemas.row_models import row_model_for_table_schema
from codeintel.build.schemas import get_schema_provider

# Generate at module load time
_provider = get_schema_provider()
FunctionMetricsRow = row_model_for_table_schema(
    table_schema=_provider.require_table_schema("analytics.function_metrics")
)
```

**Pros**:
- Minimal changes to consumers (same import paths)
- Gradual transition
- TableSchema remains source of truth

**Cons**:
- Still requires `rows/` directory
- Module-level schema access has circular import risks
- Not a true deletion of legacy code

---

## Recommended Phased Approach

### Phase 1: Audit and Categorize (PR-74)

**Goal**: Understand full scope of TypedDict usage

**Tasks**:
1. [ ] Run comprehensive import scan for all `config.datasets.rows` imports
2. [ ] Categorize imports:
   - Runtime usage (actual instantiation)
   - TYPE_CHECKING only (annotations)
   - Column constant usage
3. [ ] Document each file's migration requirements
4. [ ] Create migration complexity estimate

**Deliverable**: Complete inventory with per-file migration plan

### Phase 2: Migrate TYPE_CHECKING Imports (PR-75)

**Goal**: Move type-only imports to generated types

**Tasks**:
1. [ ] Extend `core/schemas/row_models.py` with `_TypedDictFromSchema` generator
2. [ ] Create `core/schemas/generated_types.py` with pre-generated TypedDicts
3. [ ] Update TYPE_CHECKING imports in tests and source files
4. [ ] Verify pyright/pyrefly still pass

**Deliverable**: All TYPE_CHECKING imports use generated types

### Phase 3: Migrate Runtime Consumers (PR-76)

**Goal**: Remove runtime dependency on hand-maintained TypedDicts

**Tasks**:
1. [ ] Migrate `analytics/compute/` row builders to generated types
2. [ ] Migrate `graphs/compute/` callgraph modules
3. [ ] Migrate `ingestion/compute/` typing modules
4. [ ] Migrate test helpers

**Deliverable**: All runtime row type usage migrated

### Phase 4: Migrate Column Constants (PR-77)

**Goal**: Remove dependency on `TABLE_SCHEMAS` for column constant generation

**Tasks**:
1. [ ] Refactor `_get_contract_columns()` to use `get_schema_provider()`
2. [ ] Update all column constant definitions in `rows/analytics.py`
3. [ ] Verify no circular imports
4. [ ] Verify all tests pass

**Deliverable**: `rows/analytics.py` no longer imports `TABLE_SCHEMAS`

### Phase 5: Delete Legacy Files (PR-78)

**Goal**: Remove all legacy `config/datasets/` files

**Tasks**:
1. [ ] Delete `config/datasets/schemas.py` (~114KB)
2. [ ] Delete `config/datasets/rows/` directory (~50KB)
3. [ ] Delete `config/datasets/generated_rows/` directory (~30KB)
4. [ ] Simplify `config/datasets/contracts.py` to minimal re-exports
5. [ ] Assess and potentially delete:
   - `columns.py`
   - `dataflow.py`
   - `dependencies.py`
   - `semantic_roles.py`
6. [ ] Update `config/datasets/__init__.py` with deprecation warnings
7. [ ] Remove banned-api lint rules (no longer needed)

**Deliverable**: Legacy infrastructure fully deleted

---

## File-by-File Migration Requirements

### `rows/analytics.py` (~2500 lines)

**Contains**:
- 20+ TypedDict definitions (e.g., `FunctionMetricsRow`, `CoverageLineRow`)
- 20+ serializer functions (`function_metrics_row_to_tuple`)
- 12+ column constant tuples (`_FUNCTION_METRICS_COLUMNS`)
- `_get_contract_columns()` helper

**Migration**:
1. TypedDicts → Generate from TableSchema
2. Serializers → Use `row_binding_for_table_schema().serializer`
3. Column constants → Use `schema.column_names()` directly
4. Delete `_get_contract_columns()` after migration

### `rows/core.py` (~500 lines)

**Contains**:
- Core entity TypedDicts (`GoidRow`, `ModuleRow`, `RepoMapRow`)
- Serializers for core entities

**Migration**:
- Similar to analytics.py but smaller scope

### `rows/graph.py` (~800 lines)

**Contains**:
- Graph-related TypedDicts (`CallGraphEdgeRow`, `ImportEdgeRow`, `CFGEdgeRow`)
- Serializers

**Migration**:
- Higher priority due to heavy usage in `graphs/compute/`

### `rows/profiles.py` (~300 lines)

**Contains**:
- Profile TypedDicts (`GraphMetricsFunctionsRow`, `GraphMetricsModulesRow`)

**Migration**:
- Used in test helpers and analytics

### `rows/test.py` (~400 lines)

**Contains**:
- Test-related TypedDicts (`BehavioralCoverageRowModel`, `TestCoverageEdgeRow`)

**Migration**:
- Used in test coverage and profile analytics

### `schemas.py` (~3000+ lines)

**Contains**:
- `TABLE_SCHEMAS` dictionary with all table schema definitions
- Column definitions for every table
- Index definitions

**Migration**:
- Already have `get_schema_provider()` with complete schema access
- Only blocker is `_get_contract_columns()` usage in `rows/analytics.py`
- Can delete once Phase 4 complete

### `generated_rows/` (~30KB)

**Contains**:
- Auto-generated schema-prefixed TypedDicts
- Created as migration target but rarely used

**Migration**:
- Can delete once consumers migrate to `core/schemas/generated_types.py`
- Or consolidate with new generation approach

---

## Estimated Effort

| Phase | Scope | Estimated Effort |
|-------|-------|------------------|
| Phase 1: Audit | Import scan, categorization | 1 day |
| Phase 2: TYPE_CHECKING | Type annotation migration | 2-3 days |
| Phase 3: Runtime | Row builder migration | 3-4 days |
| Phase 4: Column Constants | `rows/analytics.py` refactor | 1-2 days |
| Phase 5: Deletion | File cleanup, lint rules | 1 day |
| **Total** | | **8-11 days** |

---

## Success Criteria

The remaining scope is complete when:

1. [ ] Zero files import from `config/datasets/rows/`
2. [ ] Zero files import from `config/datasets/generated_rows/`
3. [ ] Zero files import `TABLE_SCHEMAS` directly
4. [ ] `config/datasets/schemas.py` is deleted (~114KB)
5. [ ] `config/datasets/rows/` directory is deleted (~50KB)
6. [ ] `config/datasets/generated_rows/` directory is deleted (~30KB)
7. [ ] All TypedDict row models are generated from TableSchema
8. [ ] All tests pass with generated types
9. [ ] Type checkers (pyright, pyrefly) pass
10. [ ] Total legacy code removed: ~200KB+

---

## Risks and Mitigations

### Risk: Circular imports during migration

**Mitigation**: Use lazy import patterns established in PR-69/PR-70. Separate module-level schema access from runtime access.

### Risk: Type checker failures with generated TypedDicts

**Mitigation**: Pre-generate TypedDicts at build time rather than runtime. Use `TYPE_CHECKING` guards appropriately.

### Risk: Performance regression from runtime generation

**Mitigation**: Cache generated types. Use build-time codegen for heavily-used types.

### Risk: Breaking external consumers

**Mitigation**: Keep `config/datasets/` as deprecation shim layer for one release cycle. Emit warnings before removal.

---

## Appendix: Import Scan Results

**Scan Date**: 2024-12-15

### Source Files Importing `config.datasets.rows` (8 files)

| File | Status |
|------|--------|
| `config/datasets/contracts.py` | Internal re-export, keep as shim |
| `config/datasets/__init__.py` | Internal re-export, keep as shim |
| `config/datasets/rows/__init__.py` | Self-reference, deletes with directory |
| `build/plugins/graphs/builders/callgraph.py` | **MIGRATE** — Uses `CallGraphEdgeRow` |
| `graphs/compute/callgraph/persistence.py` | **MIGRATE** — Uses `CallGraphEdgeRow` |
| `graphs/compute/callgraph/collection.py` | **MIGRATE** — Uses `CallGraphEdgeRow` |
| `analytics/compute/hotspots/metrics.py` | **MIGRATE** — Uses row models |
| `ingestion/compute/typing_ingest.py` | **MIGRATE** — Uses `TypeHintRow` |

### Test Files Importing `config.datasets.rows` (13 files)

| File | Status |
|------|--------|
| `tests/_helpers/analytics_domain.py` | **MIGRATE** — `CoverageLineRow` |
| `tests/_helpers/factories/row_factories.py` | **MIGRATE** — Various row types |
| `tests/_helpers/seeds/architecture.py` | **MIGRATE** — `GraphMetricsFunctionsRow` |
| `tests/analytics/test_analytics_contracts.py` | **MIGRATE** |
| `tests/analytics/test_profiles_and_functions.py` | **MIGRATE** |
| `tests/analytics/test_tests_profiles_unit.py` | **MIGRATE** |
| `tests/config/test_dataset_contract.py` | **MIGRATE** |
| `tests/docs_export/test_graph_validation_export.py` | **MIGRATE** |
| `tests/graphs/test_callgraph_resolution.py` | **MIGRATE** |
| `tests/graphs/test_compute_layer.py` | **MIGRATE** |
| `tests/storage/test_file_module_test_behavioral_columns.py` | **MIGRATE** |
| `tests/storage/test_function_profile_rows.py` | **MIGRATE** |
| `tests/storage/test_schema_roundtrip.py` | **MIGRATE** |

### Files Importing `config.datasets.generated_rows` (2 source files)

| File | Status |
|------|--------|
| `storage/gateway/accessors.py` | **MIGRATE** — Uses generated row types |
| `tests/storage/test_gateway_accessors.py` | **MIGRATE** — Test for above |

### Files Importing `TABLE_SCHEMAS` Directly (1 source file)

| File | Status |
|------|--------|
| `config/datasets/rows/analytics.py` | **BLOCKING** — Uses `_get_contract_columns()` |

### Summary

| Category | Count | Effort |
|----------|-------|--------|
| Source files with `rows/` imports | 5 (excl. internal) | Medium |
| Test files with `rows/` imports | 13 | Low (TYPE_CHECKING mostly) |
| Source files with `generated_rows/` imports | 1 | Low |
| Test files with `generated_rows/` imports | 1 | Low |
| Files with `TABLE_SCHEMAS` imports | 1 | High (blocking) |
| **Total files to migrate** | **21** | |

To re-run the scan:

```bash
rg "from codeintel.config.datasets.rows" --type py -l
rg "from codeintel.config.datasets.generated_rows" --type py -l
rg "from codeintel.config.datasets.schemas import TABLE_SCHEMAS" --type py -l
```

---

## Related Documents

- `Hamilton_consolidation_phase3.md` — Completed work (PR-66 through PR-73)
- `AGENTS.md` — Development standards and patterns

