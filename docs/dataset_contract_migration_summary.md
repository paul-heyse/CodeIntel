# Dataset Contract Modularization — Migration Summary

> **Status**: Phases 2-5 Complete — Schemas Migrated  
> **Date**: December 2, 2025  
> **Epic**: data_integration_epic_6.md

---

## Executive Summary

The monolithic `src/codeintel/config/dataset_contract.py` (5,395 lines) has been modularized into a new `codeintel.config.datasets` package. All 90 files across the codebase now import from the new package.

**Key accomplishments:**
- `TABLE_SCHEMAS` (79 schemas, ~1,845 lines) moved to `schemas.py`
- `COMPOSITE_SCHEMAS` (4 schemas, ~322 lines) moved to `schemas.py`
- `rows/` package structure created for TypedDict row models
- All external imports use `codeintel.config.datasets`
- Legacy `_build_contracts()` updated to use new schema imports for object identity

The remaining definitions (row TypedDicts, metadata dictionaries, ROW_BINDINGS, _build_contracts) are accessed via lazy imports from the legacy file during the migration period.

---

## Current Package Structure

```
src/codeintel/config/datasets/
├── __init__.py      # Unified facade with lazy loading
├── primitives.py    # Core types: Column, Index, TableSchema, CompositeSchema, column fragments
├── schemas.py       # TABLE_SCHEMAS (79), COMPOSITE_SCHEMAS (4) [MIGRATED]
├── contracts.py     # DatasetContract, RowBinding, accessor functions
├── rows/            # Package for row TypedDicts and serializers
│   └── __init__.py  # Re-exports from legacy during migration
├── sql.py           # SQL generation helpers (INSERT, DELETE)
└── dataflow.py      # DataflowNode, DataflowEdge, graph builders
```

---

## Completed Phases

### ✅ Phase 1: API Facade Established (Prior Work)

- Created `codeintel.config.datasets` package
- Updated 90 files to use new import paths
- Defined `primitives.py` with core types and column fragments

### ✅ Phase 2: TABLE_SCHEMAS Migrated

**File**: `schemas.py`

- 79 table/view schemas (~1,845 lines) moved to `schemas.py`
- All external code imports via `codeintel.config.datasets`
- `contracts.py` accessor functions import from `schemas.py`

### ✅ Phase 3: COMPOSITE_SCHEMAS Migrated

**File**: `schemas.py`

- 4 composite schemas (~322 lines) moved to `schemas.py`
- `_FUNCTION_PROFILE_ENTITY_COLS` constant included
- Profile datasets correctly reference schemas

### ✅ Phase 4: Row Package Structure Created

**Files**: `rows/__init__.py`

- Created `rows/` subpackage for TypedDict row models
- Currently re-exports 37 types and 35 functions from legacy
- Ready for incremental migration

### ✅ Phase 5: Contract Building Uses New Schemas

**File**: `dataset_contract.py` (updated)

- `_build_contracts()` imports TABLE_SCHEMAS and COMPOSITE_SCHEMAS from `schemas.py`
- Ensures object identity for `DatasetContract.composition` references
- All tests pass including composition identity checks

---

## Quality Gates Verified

| Check | Result |
|-------|--------|
| Pyright | 0 errors, 0 warnings |
| Pyrefly | 0 errors (3 acceptable redundant cast warnings) |
| Ruff | All checks passed |
| Config tests | 110 passing |
| Snapshot counts | TABLE_SCHEMAS=79, COMPOSITE_SCHEMAS=4, DATASET_CONTRACTS=108, ROW_BINDINGS=36 |

---

## Roadmap to Full Migration (Phase 6)

The following work remains to fully eliminate backward compatibility code and delete the legacy file.

### Phase 6a: Move Row TypedDicts and Serializers (~1,530 lines)

**Current State**: Row types re-exported from legacy via `rows/__init__.py`

**Target Structure**:
```
rows/
├── __init__.py       # Re-exports all symbols
├── core.py           # IngestRunRow, GoidRow, GoidCrosswalkRow, DocstringRow, ConfigValueRow
├── analytics.py      # CoverageLineRow, TypednessRow, FunctionMetricsRow, FunctionTypesRow,
│                     # StaticDiagnosticRow, HotspotRow, FunctionValidationRow, GraphValidationRow
├── graph.py          # CallGraphNodeRow, CallGraphEdgeRow, ImportEdgeRow, ImportModuleRow,
│                     # CFGBlockRow, CFGEdgeRow, DFGEdgeRow, SymbolUseRow
├── profiles.py       # FunctionProfileRowModel, FileProfileRowModel, ModuleProfileRowModel,
│                     # GraphMetricsFunctionsRow, GraphMetricsModulesRow, etc.
└── test.py           # TestCatalogRowModel, TestCoverageEdgeRow, ProfileRowModel,
                      # BehavioralCoverageRowModel, SubsystemProfileCacheRow, etc.
```

**Work Required**:
1. Extract 37 TypedDict definitions from legacy (lines 2828-4354)
2. Extract 35 serializer functions
3. Extract 14 column constant tuples (`FUNCTION_METRICS_COLUMNS`, etc.)
4. Move `_serialize_row()` and `_get_contract_columns()` helpers to `sql.py`
5. Update `rows/__init__.py` to import from submodules

**Estimated Effort**: 4-6 hours

---

### Phase 6b: Move Metadata Dictionaries (~350 lines)

**Current State**: Metadata dictionaries in legacy, accessed via lazy imports

**Dictionaries to Move to `contracts.py`**:
- `_JSON_SCHEMA_BY_DATASET_NAME` (~20 entries)
- `_DESCRIPTION_BY_DATASET_NAME` (~15 entries)
- `_OWNER_BY_DATASET_NAME` (~15 entries)
- `_FRESHNESS_BY_DATASET_NAME` (~15 entries)
- `_RETENTION_BY_DATASET_NAME` (~15 entries)
- `_STABLE_ID_BY_DATASET_NAME` (empty)
- `_SCHEMA_VERSION_BY_DATASET_NAME` (empty)
- `_VALIDATION_PROFILE_BY_DATASET_NAME` (empty)
- `_DEPENDENCIES_BY_DATASET_NAME` (~25 entries)
- `_DEFAULT_JSONL_FILENAMES` (~80 entries)
- `_DEFAULT_PARQUET_FILENAMES` (~80 entries)
- `_DATASET_ROWS_ONLY` (~35 entries)

**Helper Functions to Move**:
- `_metadata_for_name()`
- `_owner_package_for_prefix()`

**Estimated Effort**: 1 hour

---

### Phase 6c: Move ROW_BINDINGS_BY_TABLE_KEY (~150 lines)

**Current State**: Lazy-loaded from legacy in `contracts.py`

**Work Required**:
1. Move `_row_binding()` helper to `contracts.py`
2. Move the full `ROW_BINDINGS_BY_TABLE_KEY` dictionary (36 entries)
3. Update imports to use `rows/` submodules
4. Remove lazy loading from `get_row_bindings()`

**Dependency**: Requires Phase 6a (rows migration)

**Estimated Effort**: 1 hour

---

### Phase 6d: Move `_build_contracts()` Function (~80 lines)

**Current State**: In legacy file, but already imports from `schemas.py`

**Work Required**:
1. Move `_build_contracts()` to `contracts.py`
2. Update to use local metadata dictionaries and ROW_BINDINGS
3. Build `DATASET_CONTRACTS` and `DATASET_CONTRACTS_BY_TABLE_KEY` locally
4. Remove lazy loading from accessor functions

**Dependencies**: Requires Phases 6a-6c

**Estimated Effort**: 30 minutes

---

### Phase 6e: Remove Legacy Constants from `__init__.py`

**Current State**: Facade forwards some legacy constants via `_get_legacy_constant()`

**Work Required**:
1. Remove `_get_legacy_constant()` function
2. Remove any remaining legacy constant forwarding
3. Update `__all__` to reflect local definitions only

**Estimated Effort**: 15 minutes

---

### Phase 6f: Delete Legacy File

**Prerequisites**:
- All phases 6a-6e complete
- Zero imports from `codeintel.config.dataset_contract`

**Verification**:
```bash
# Should return empty
grep -rn "from codeintel.config.dataset_contract import" src/ tests/
```

**Work Required**:
1. Delete `src/codeintel/config/dataset_contract.py`
2. Run full test suite: `uv run pytest -q`
3. Verify quality gates: `uv run python -m tools.quality_report`

**Estimated Effort**: 15 minutes

---

## Total Remaining Effort

| Phase | Description | Effort |
|-------|-------------|--------|
| 6a | Row TypedDicts and serializers | 4-6 hours |
| 6b | Metadata dictionaries | 1 hour |
| 6c | ROW_BINDINGS_BY_TABLE_KEY | 1 hour |
| 6d | `_build_contracts()` | 30 min |
| 6e | Remove legacy constants | 15 min |
| 6f | Delete legacy file | 15 min |
| **Total** | | **~7-9 hours** |

---

## Go-Forward Architecture (Final State)

After completing all Phase 6 subphases:

```
src/codeintel/config/datasets/
├── __init__.py        # Clean facade, no legacy imports
├── primitives.py      # Core types: Column, Index, TableSchema, CompositeSchema
├── schemas.py         # TABLE_SCHEMAS (79), COMPOSITE_SCHEMAS (4)
├── contracts.py       # DatasetContract, RowBinding, _build_contracts(), all metadata
├── rows/
│   ├── __init__.py    # Re-exports all row types and serializers
│   ├── core.py        # IngestRunRow, GoidRow, etc.
│   ├── analytics.py   # FunctionMetricsRow, CoverageLineRow, etc.
│   ├── graph.py       # CallGraphNodeRow, CFGBlockRow, etc.
│   ├── profiles.py    # FunctionProfileRowModel, etc.
│   └── test.py        # TestCatalogRowModel, etc.
├── sql.py             # SQL generation helpers, _serialize_row()
└── dataflow.py        # DataflowNode, DataflowEdge, graph builders

# DELETED: src/codeintel/config/dataset_contract.py (was 5,395 lines)
```

**Benefits of Final Architecture**:
- **Modularity**: Each file has a single responsibility
- **Maintainability**: Row types organized by domain
- **Import Performance**: No lazy loading required
- **Type Safety**: Clean type definitions without casts
- **Discoverability**: Intuitive package structure

---

## Current Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    External Code (90 files)                      │
│         from codeintel.config.datasets import ...                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              codeintel.config.datasets/__init__.py               │
│   • Re-exports all symbols from submodules                       │
│   • Lazy loading for contracts/bindings from legacy              │
│   • Eager imports for schemas.py (TABLE_SCHEMAS, COMPOSITE)      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┬───────────────┐
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ primitives  │   │  schemas    │   │  contracts  │   │   rows/     │
│ .py         │   │  .py        │   │  .py        │   │ __init__.py │
│             │   │             │   │             │   │             │
│ Column      │   │ TABLE_      │   │ Dataset-    │   │ Re-exports  │
│ Index       │   │ SCHEMAS     │   │ Contract    │   │ from legacy │
│ TableSchema │   │ (79)        │   │ RowBinding  │   │ (37 types)  │
│ Composite-  │   │             │   │             │   │ (35 funcs)  │
│ Schema      │   │ COMPOSITE_  │   │ Accessors   │   │             │
│ Column      │   │ SCHEMAS (4) │   │ for lazy    │   │             │
│ fragments   │   │             │   │ loading     │   │             │
└─────────────┘   └─────────────┘   └──────┬──────┘   └──────┬──────┘
                                           │                  │
                                           ▼                  ▼
                       ┌────────────────────────────────────────────┐
                       │   dataset_contract.py (LEGACY)              │
                       │   Remaining content (~2,000 lines):         │
                       │   • Row TypedDicts (37 types)               │
                       │   • Serializers (35 functions)              │
                       │   • ROW_BINDINGS_BY_TABLE_KEY (36 entries)  │
                       │   • _build_contracts() logic                │
                       │   • Metadata dictionaries (12 dicts)        │
                       │                                             │
                       │   (TABLE_SCHEMAS, COMPOSITE_SCHEMAS         │
                       │    already migrated to schemas.py)          │
                       └────────────────────────────────────────────┘
```

---

## Migration Commands Reference

```bash
# Check remaining legacy imports
grep -rn "from codeintel.config.dataset_contract import" src/ tests/

# Run tests for the datasets package
uv run pytest tests/config/test_dataset* -v

# Type check the package
uv run pyright src/codeintel/config/datasets/ --warnings --pythonversion=3.13
uv run pyrefly check src/codeintel/config/datasets/

# Lint check
uv run ruff check src/codeintel/config/datasets/ --fix

# Verify counts haven't changed
uv run python -c "
from codeintel.config.datasets import TABLE_SCHEMAS, DATASET_CONTRACTS, ROW_BINDINGS_BY_TABLE_KEY, COMPOSITE_SCHEMAS
print(f'TABLE_SCHEMAS: {len(TABLE_SCHEMAS)}')  # Expected: 79
print(f'COMPOSITE_SCHEMAS: {len(COMPOSITE_SCHEMAS)}')  # Expected: 4
print(f'DATASET_CONTRACTS: {len(DATASET_CONTRACTS)}')  # Expected: 108
print(f'ROW_BINDINGS: {len(ROW_BINDINGS_BY_TABLE_KEY)}')  # Expected: 36
"

# Full quality report
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
```

---

## Files Modified in This Migration

### New files created (Phase 1):
- `src/codeintel/config/datasets/__init__.py`
- `src/codeintel/config/datasets/primitives.py`
- `src/codeintel/config/datasets/contracts.py`
- `src/codeintel/config/datasets/sql.py`
- `src/codeintel/config/datasets/dataflow.py`
- `tests/config/test_dataset_contract_snapshot.py`
- `tests/config/test_datasets_primitives.py`
- `tests/config/test_datasets_contracts.py`
- `tests/config/test_datasets_sql.py`
- `tests/config/test_datasets_dataflow.py`

### New files created (Phases 2-5):
- `src/codeintel/config/datasets/schemas.py` — TABLE_SCHEMAS and COMPOSITE_SCHEMAS
- `src/codeintel/config/datasets/rows/__init__.py` — Row type re-exports

### Files with updated imports (90 files):
See `git diff --name-only` for the full list, primarily in:
- `src/codeintel/graphs/`
- `src/codeintel/ingestion/`
- `src/codeintel/pipeline/`
- `src/codeintel/serving/`
- `src/codeintel/storage/`
- `src/codeintel/analytics/`
- `tests/`

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Snapshot count drift | `test_dataset_contract_snapshot.py` locks counts |
| Circular imports | Lazy loading with `__getattr__` |
| Type compatibility | Explicit `cast()` during migration |
| Missing exports | Comprehensive `__all__` in `__init__.py` |
| Object identity | Legacy `_build_contracts()` imports from `schemas.py` |

---

## Related Documentation

- **Epic plan**: `docs/data_integration_epic_6.md`
- **Prior epics**: `docs/Data_integration_epic_3.md`, `epic_4.md`, `epic_5.md`
- **Migration plan**: `cursor-plan://Dataset Contract Modularization.plan.md`
