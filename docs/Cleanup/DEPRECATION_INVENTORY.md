# Deprecation and Legacy Code Inventory

> **Purpose**: Complete inventory of all deprecation, legacy, compatibility, and backward-compat patterns identified in `analytics`, `ingestion`, and `graphs` modules.
>
> **Created**: 2024-12-14
> **Status**: Phase A cleanup complete, remaining items documented below

---

## Overview

This document catalogs every instance of deprecated, legacy, or backward-compatibility code patterns found in the target modules. Items are categorized by action priority.

### Legend

| Status | Meaning |
|--------|---------|
| ✅ Deleted | Removed in Phase A |
| 🔴 Delete | Should be deleted (no consumers) |
| 🟡 Migrate | Has consumers that need migration first |
| 🟢 Keep | Intentional compatibility layer, keep |
| 📝 Update | Needs docstring/label update only |

---

## analytics/ Module

### Deleted in Phase A

| File | Item | Status | Notes |
|------|------|--------|-------|
| `analytics/ports/__init__.py` | Entire file | ✅ Deleted | Re-exported deprecated items from graphs.ports, no consumers |

### Re-export Modules (Deleted in Phase B)

| File | Pattern | Status | Notes |
|------|---------|--------|-------|
| `analytics/parsing/models.py` | Re-exports `SourceSpan`, `ParsedFunction`, `ParsedModule` from `core.parsing` | ✅ Deleted | File deleted, consumers migrated to `core.parsing` |
| `analytics/resources/graphs.py` | `GraphResources` alias for `GraphBundle` | ✅ Deleted | Alias removed, replaced with `GraphBundle` directly |
| `analytics/compute/graphs/conversions.py` | Re-exports `normalize_decimal_id` from `core.data_models.ids` | ✅ Deleted | Re-export removed, consumers migrated to `core.data_models.ids` |
| `analytics/compute/functions/typedness.py` | Re-exports typedness utilities | 🟢 Keep | Original code, NOT a re-export module |
| `analytics/cfg_dfg/cfg_core.py` | `function_metadata = load_function_metadata` | ✅ Deleted | Alias removed |
| `analytics/cfg_dfg/dfg_core.py` | `dfg_function_metadata = load_function_metadata` | ✅ Deleted | Alias removed |
| `analytics/profiles/utils.py` | Re-exports type coercion helpers | ✅ Deleted | Re-exports removed, consumers migrated to `utilities.type_coercion` |

### Deprecated Parameters (Removed in Phase C)

| File | Item | Status | Notes |
|------|------|--------|-------|
| `analytics/subsystems/materialize.py` | `engine` parameter | ✅ Deleted | Removed param, docstring, fallback logic, and `GraphEngine` import |

### Type Aliases

| File | Item | Status | Notes |
|------|------|--------|-------|
| `analytics/compute/row_builders/symbol_metrics.py` | `SymbolModuleMetricInputs`, `SymbolFunctionMetricInputs` | 📝 Updated | Comment updated from "backward compat" to "convenience aliases" - these are active type aliases, not deprecated |

### Test/Internal Dead Code (Deleted in Phase D)

| File | Item | Status | Notes |
|------|------|--------|-------|
| `analytics/testing/profiles/rows.py:68` | `PreparedStatements` class | ✅ Deleted | Dead code - only used by `prepared_statements_dynamic()` which was never called |
| `analytics/testing/profiles/rows.py:101` | `prepared_statements_dynamic()` | ✅ Deleted | Dead code - never called in production, only mock-target in tests |

### Comments/Labels Only (No Code Change)

| File | Item | Notes |
|------|------|-------|
| `analytics/runtime/context.py:4` | Comment: "This is NOT the deprecated plugin runtime infrastructure" | Clarifying comment only |
| `analytics/compute/functions/goids.py:6` | Comment: "without the deprecated adapter layer" | Historical reference |
| `analytics/functions/function_history.py:188` | String literal `"legacy_hot"` | Data value, not deprecation |
| `analytics/utilities/datasets.py:359` | "DuckDB compatibility" | Technical compatibility, not deprecation |

---

## ingestion/ Module

### Re-export Modules (Cleaned in Phase E)

| File | Pattern | Status | Notes |
|------|---------|--------|-------|
| `ingestion/infrastructure/ast_utils.py` | Re-exported `AstSpanIndex` from `core.parsing` | ✅ Deleted | Re-export removed, consumer migrated to `core.parsing` |
| `ingestion/infrastructure/db_queries.py` | Re-exported query utilities from `storage.queries.safe` | ✅ Deleted | File deleted, consumers migrated to `storage.queries.safe` |

### Docstring Updates (Completed in Phase E)

| File | Item | Status | Notes |
|------|------|--------|-------|
| `ingestion/adapters/duckdb_storage.py:1-7` | Docstring said "compatibility shim" | ✅ Updated | Enhanced with "Why This Adapter Exists" section explaining testability value |
| `ingestion/adapters/duckdb_storage.py:45` | Class docstring said "Compatibility shim" | ✅ Updated | Enhanced with full explanation of testability pattern and examples |
| `ingestion/ports/storage.py:179` | `IngestStoragePort` protocol docstring | ✅ Enhanced | Added "Why This Abstraction Exists" section, cross-references to implementations |
| `tests/_helpers/fakes/storage.py:22` | `FakeIngestStorage` docstring | ✅ Enhanced | Added "Why This Exists" section linking to protocol and production adapter |

### Interface Compatibility Comments (Keep)

| File | Item | Status | Notes |
|------|------|--------|-------|
| `ingestion/adapters/build_tool_adapter.py:38` | "IngestToolPort-compatible interface" | 🟢 Keep | Accurate interface description |
| `ingestion/adapters/build_tool_adapter.py:123,149,153,211-213` | "interface compatibility" in docstrings | 🟢 Keep | Accurate param documentation |
| `ingestion/engine/plugins.py:22-23,259-260` | "compatible with `AsyncPluginProtocol`" | 🟢 Keep | Accurate protocol compatibility |
| `ingestion/engine/plugins.py:408,424` | "core-compatible metadata" | 🟢 Keep | Describes conversion to core format |

### Ingestion Ports Layer

| File | Item | Status | Notes |
|------|------|--------|-------|
| `ingestion/ports/storage.py:4` | "backward-compatible aliases for the ingestion naming convention" | 🟢 Keep | Legitimate abstraction layer |
| `ingestion/ports/storage.py:38,53-54,64,75,112-113,126-127,149,169` | Various "compatibility" comments | 🟢 Keep | Documents dual naming convention (ingestion vs core) |

---

## graphs/ Module

### Deleted in Phase A

| File | Item | Status | Notes |
|------|------|--------|-------|
| `graphs/resources/catalog.py` | `CatalogResource` alias | ✅ Deleted | Deprecated wrapper for CatalogService |
| `graphs/ports/catalog.py` | `CatalogPort` protocol | ✅ Deleted | Deprecated protocol |
| `graphs/ports/catalog.py` | `FunctionSpanData` compat class | ✅ Deleted | Deprecated alias for FunctionSpan |
| `graphs/ports/catalog.py` | `_FunctionSpanDataCompat` | ✅ Deleted | Implementation class for above |
| `graphs/ports/catalog.py` | `FunctionSpanDataType` | ✅ Deleted | Type alias |
| `graphs/ports/engine.py` | `EnginePort` protocol | ✅ Deleted | Deprecated protocol |
| `graphs/ports/storage.py` | `StoragePort` re-export | ✅ Deleted | Removed from exports |
| `graphs/ports/__init__.py` | Deprecated protocol exports | ✅ Deleted | Cleaned up exports |
| `graphs/resources/__init__.py` | `CatalogResource` export | ✅ Deleted | Removed from exports |

### Deprecated Aliases (Deleted in Phase A-Extended)

| File | Item | Status | Notes |
|------|------|--------|-------|
| `graphs/catalog.py` | `FunctionMeta` compat class | ✅ Deleted | Was emitting DeprecationWarning |
| `graphs/catalog.py` | `_FunctionMetaCompat` class | ✅ Deleted | Implementation for FunctionMeta |
| `graphs/catalog.py` | `_create_function_meta()` function | ✅ Deleted | Helper for FunctionMeta |
| `graphs/catalog.py` | `FunctionCatalogService()` function | ✅ Deleted | Was emitting DeprecationWarning |
| `graphs/catalog.py` | `FunctionCatalogServiceType` type alias | ✅ Deleted | Alias for CatalogService |
| `graphs/catalog.py` | `FunctionMeta` in `__all__` | ✅ Deleted | Removed from exports |
| `graphs/catalog.py` | `FunctionCatalogService` in `__all__` | ✅ Deleted | Removed from exports |
| `graphs/catalog.py:58-61` | `SpanIndex` wrapper class docstring | 🟢 Keep | Documents it's for backward compat but still useful |

### Backward-Compatible Function Wrappers (Keep by Design)

These are intentionally retained for API stability:

| File | Functions | Status | Notes |
|------|-----------|--------|-------|
| `graphs/validation/checks/structure.py:641-844` | `call_graph_findings`, `import_graph_findings`, `import_cycle_findings`, `import_hub_findings`, `import_upward_findings`, `import_bridge_findings`, `symbol_graph_findings`, `config_key_findings` | 🟢 Keep | Wrappers around CheckProtocol classes for API stability |
| `graphs/validation/checks/database.py:340-447` | `warn_missing_function_goids`, `warn_callsite_span_mismatches`, `warn_orphan_modules`, `warn_graph_structure` | 🟢 Keep | Wrappers around CheckProtocol classes |
| `graphs/validation/checks/anomaly.py:184-227` | `symbol_community_findings`, `subsystem_disagreement_findings` | 🟢 Keep | Wrappers around CheckProtocol classes |
| `graphs/validation/__init__.py` | Re-exports above functions | 🟢 Keep | API stability exports |
| `graphs/validation/checks/__init__.py` | Re-exports above functions | 🟢 Keep | API stability exports |
| `graphs/validation/runner.py:13-14` | "Legacy function-based validation" mode | 🟢 Keep | Supports both old and new patterns |

### Re-export Modules

| File | Pattern | Status | Notes |
|------|---------|--------|-------|
| `graphs/catalog.py:36-37` | Re-exports `FunctionSpan` from `core.catalog` | 🟢 Keep | Core type re-export |
| `graphs/catalog.py:38` | Re-exports `SpanIndex` from `core.catalog` | 🟢 Keep | Core type re-export |
| `graphs/compute/metrics/centrality.py:4` | Re-exports from `core.compute.centrality` | 🟢 Keep | Good migration pattern |

### Other Comments

| File | Item | Notes |
|------|------|-------|
| `graphs/engine/views.py:79` | "retained for backwards compatibility" | Parameter naming compatibility |
| `graphs/compute/metrics/coupling.py:111` | "kept for signature compatibility" | Parameter documentation |

---

## Summary Statistics

### By Status

| Status | Count | Description |
|--------|-------|-------------|
| ✅ Deleted/Updated | 31 | Removed/updated in Phase A through E |
| 🔴 Delete | 0 | Ready to delete (no/minimal consumers) |
| 🟡 Migrate | 0 | Needs consumer migration first |
| 🟢 Keep | 25+ | Intentional compatibility, keep |
| 📝 Update | 0 | All docstring updates complete |

### By Module

| Module | Total Items | Deleted | To Delete | To Migrate | Keep | Update |
|--------|-------------|---------|-----------|------------|------|--------|
| analytics/ | 15 | 10 | 0 | 0 | 4 | 1 |
| ingestion/ | 12 | 4 | 0 | 0 | 8 | 0 |
| graphs/ | 30+ | 17 | 0 | 0 | 13+ | 0 |

---

## Action Plan

### Phase A (Complete ✅)
- Deleted `analytics/ports/` package
- Cleaned up `graphs/ports/` deprecated protocols
- Deleted `graphs/resources/catalog.py`

### Phase A-Extended (Complete ✅)
- Deleted `FunctionMeta`, `_FunctionMetaCompat`, `_create_function_meta()` from `graphs/catalog.py`
- Deleted `FunctionCatalogService()`, `FunctionCatalogServiceType` from `graphs/catalog.py`
- Updated `__all__` exports

### Phase B (Complete ✅)
Removed analytics re-export modules and aliases:
- **Deleted** `analytics/parsing/models.py` (consumers migrated to `core.parsing`)
- **Removed** `GraphResources` alias from `analytics/resources/graphs.py` (now uses `GraphBundle` directly)
- **Removed** `normalize_decimal_id` re-export from `analytics/compute/graphs/conversions.py` (consumers migrated to `core.data_models.ids`)
- **Removed** `function_metadata` alias from `analytics/cfg_dfg/cfg_core.py`
- **Removed** `dfg_function_metadata` alias from `analytics/cfg_dfg/dfg_core.py`
- **Removed** type coercion re-exports from `analytics/profiles/utils.py` (consumers migrated to `utilities.type_coercion`)

### Phase C (Complete ✅)
Removed deprecated `engine` parameter:
- **Removed** `engine` param from `build_subsystems()` signature in `analytics/subsystems/materialize.py`
- **Removed** docstring entry for `engine`
- **Removed** fallback logic that handled the deprecated parameter
- **Removed** unused `GraphEngine` import

### Phase D (Complete ✅)
Cleaned up dead test code and updated type alias comments:
- **Deleted** `PreparedStatements` class from `analytics/testing/profiles/rows.py` (dead code, never used)
- **Deleted** `prepared_statements_dynamic()` from `analytics/testing/profiles/rows.py` (dead code, never called)
- **Removed** unused imports (`sqlglot.expressions`, `get_dataset_contracts_by_table_key`)
- **Updated** test overrides in `tests/analytics/test_tests_profiles_unit.py` to remove mock of deleted function
- **Updated** comment in `analytics/compute/row_builders/symbol_metrics.py` from "backward compatibility" to "convenience aliases"

### Phase E (Complete ✅)
Cleaned up ingestion re-exports and docstrings:
- **Removed** `AstSpanIndex` re-export from `ingestion/infrastructure/ast_utils.py` (consumer migrated to `core.parsing`)
- **Deleted** `ingestion/infrastructure/db_queries.py` (pure re-export file, consumers migrated to `storage.queries.safe`)
- **Updated** module docstring in `ingestion/adapters/duckdb_storage.py` (removed "shim" terminology)
- **Updated** class docstring in `DuckDBStorageAdapter` (removed "Compatibility shim" terminology)

### Phase F (Optional - Low Priority)
1. Review all 🟢 Keep items for potential future consolidation
2. Consider whether remaining re-export patterns should emit deprecation warnings

---

## Appendix: Search Patterns Used

```bash
# Primary search pattern
grep -r "deprecat|legacy|compat|backward|shim|obsolete|TODO.*migrat" \
  src/codeintel/{analytics,ingestion,graphs}/ -i

# Consumer search patterns
grep -r "from codeintel\.X import" src/ tests/
grep -r ": ProtocolName" src/  # Type annotation usage
```

---

## Change Log

| Date | Change |
|------|--------|
| 2024-12-14 | Initial inventory created |
| 2024-12-14 | Phase A complete - marked 11 items as deleted |
| 2024-12-14 | Phase A-Extended complete - deleted 7 more items from graphs/catalog.py (total: 18 deleted) |
| 2024-12-14 | Phase B complete - deleted 6 analytics re-export items (total: 24 deleted) |
| 2024-12-14 | Phase C complete - removed deprecated `engine` param from `build_subsystems()` (total: 25 deleted) |
| 2024-12-14 | Phase D complete - deleted dead test code (`PreparedStatements`, `prepared_statements_dynamic`), updated type alias comment (total: 27 deleted) |
| 2024-12-14 | Phase E complete - cleaned up ingestion re-exports (`AstSpanIndex`, `db_queries.py`), updated `duckdb_storage.py` docstrings (total: 31 deleted/updated) |
| 2024-12-14 | Phase E addendum - enhanced docstrings for `IngestStoragePort`, `DuckDBStorageAdapter`, `FakeIngestStorage` with "Why This Exists" documentation |
