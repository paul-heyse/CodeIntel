# Adapters Migration Plan: Transition to Hamilton-Native Patterns

**Document Version:** 2.0  
**Created:** 2025-12-13  
**Last Updated:** 2025-12-13  
**Status:** Part 1 Complete, Parts 2-3 In Planning  
**Scope:** `analytics/adapters/`, `graphs/adapters/`, `ingestion/adapters/`

---

## Completed Work Summary

### Part 1: Analytics Adapters - COMPLETE

**Status:** Fully implemented and verified on 2025-12-13

#### What Was Done

| Work Package | Description | LOC Change |
|--------------|-------------|------------|
| **WP-1: Extract DeleteScope** | Moved to `analytics/utilities/persistence.py` | +35 |
| **WP-2: Move Row Builders** | `adapters/graphs/` → `compute/row_builders/` | +200 (reorganized) |
| **WP-3: Extract Data Types** | Created `config/datasets/dependencies.py` (~160 LOC) | +160 |
| **WP-4: Extract GOID Types** | Created `compute/functions/goids.py` (~200 LOC) | +200 |
| **WP-5: Extract Semantic Roles** | Created `config/datasets/semantic_roles.py` (~340 LOC) | +340 |
| **WP-6: Delete Adapters** | Removed 9 adapter files, ~3,100 LOC | -3,100 |
| **WP-7: Delete Tests** | Removed 10 test files, ~3,560 LOC | -3,560 |
| **WP-8: Error Stub** | Created helpful `__init__.py` with migration guidance | +65 |

**Net Result:** -5,971 LOC removed

#### Files Created
```
src/codeintel/analytics/utilities/persistence.py      # DeleteScope
src/codeintel/config/datasets/dependencies.py         # DependencyCallRow, DependencyAggregateRow
src/codeintel/config/datasets/semantic_roles.py       # FunctionSemanticRoleRow, ModuleSemanticRoleRow
src/codeintel/analytics/compute/functions/goids.py    # FunctionGoid, FunctionGoidLoader, GoidRow
src/codeintel/analytics/compute/row_builders/         # Moved from adapters/graphs/
├── __init__.py
├── graph_metrics.py
├── graph_metrics_ext.py
├── subsystem_metrics.py
└── symbol_metrics.py
```

#### Files Deleted
```
src/codeintel/analytics/adapters/
├── base.py                  # Deleted
├── data_models.py           # Deleted
├── dependencies.py          # Deleted (types extracted)
├── entrypoints.py           # Deleted
├── functions.py             # Deleted (types extracted)
├── profiles.py              # Deleted
├── schema_adapter.py        # Deleted
├── semantic_roles.py        # Deleted (types extracted)
├── subsystems.py            # Deleted
└── graphs/                  # Deleted (moved to compute/)

tests/analytics/adapters/    # Entire directory deleted (~3,560 LOC)
```

#### Migration Paths for Old Imports

| Old Import | New Import |
|------------|------------|
| `analytics.adapters.base.DeleteScope` | `analytics.utilities.persistence.DeleteScope` |
| `analytics.adapters.dependencies.DependencyCallRow` | `config.datasets.dependencies.DependencyCallRow` |
| `analytics.adapters.dependencies.compute_dep_id` | `config.datasets.dependencies.compute_dep_id` |
| `analytics.adapters.functions.FunctionGoid` | `analytics.compute.functions.goids.FunctionGoid` |
| `analytics.adapters.functions.FunctionGoidLoader` | `analytics.compute.functions.goids.FunctionGoidLoader` |
| `analytics.adapters.semantic_roles.FunctionSemanticRoleRow` | `config.datasets.semantic_roles.FunctionSemanticRoleRow` |
| `analytics.adapters.graphs.*` (row builders) | `analytics.compute.row_builders.*` |

#### Verification
- All imports from new locations work correctly
- Old imports raise helpful `ImportError` with migration guidance
- `pyright` and `pyrefly` pass with no errors
- All affected tests pass

---

## Executive Summary

This document outlines a phased migration plan for transitioning adapter implementations across three domain packages to Hamilton-native patterns. The adapters were designed for the legacy plugin orchestration system and represent ~4,800+ LOC that can be consolidated, simplified, or eliminated as the Hamilton build system matures.

### Package Overview

| Package | Original | Current | Status | Notes |
|---------|----------|---------|--------|-------|
| `analytics/adapters/` | 11 files, ~3,100 LOC | 1 file, ~65 LOC | ✅ **COMPLETE** | -5,971 LOC removed |
| `graphs/adapters/` | 5 files, ~950 LOC | 5 files, ~950 LOC | 🔄 **PENDING** | ~700 LOC deletable |
| `ingestion/adapters/` | 6 files, ~1,600 LOC | 6 files, ~1,600 LOC | ⏸️ **NO CHANGES** | Actively used |

**Total Reduction:** ~6,670 LOC (after Parts 1-2 complete)

---

## Part 1: Analytics Adapters - COMPLETE ✅

**Status:** Fully implemented on 2025-12-13

### Final State

```
src/codeintel/analytics/adapters/
└── __init__.py           (65 LOC)  - Migration error stub only

# Data types extracted to:
src/codeintel/analytics/utilities/persistence.py    # DeleteScope
src/codeintel/config/datasets/dependencies.py       # DependencyCallRow, etc.
src/codeintel/config/datasets/semantic_roles.py     # FunctionSemanticRoleRow, etc.
src/codeintel/analytics/compute/functions/goids.py  # FunctionGoid, FunctionGoidLoader

# Row builders moved to:
src/codeintel/analytics/compute/row_builders/
├── __init__.py
├── graph_metrics.py
├── graph_metrics_ext.py
├── subsystem_metrics.py
└── symbol_metrics.py
```

### What Was Done

1. **Extracted DeleteScope** → `analytics/utilities/persistence.py`
2. **Moved row builders** → `analytics/compute/row_builders/`
3. **Extracted data types:**
   - `DependencyCallRow`, `DependencyAggregateRow` → `config/datasets/dependencies.py`
   - `FunctionGoid`, `FunctionGoidLoader`, `GoidRow` → `compute/functions/goids.py`
   - `FunctionSemanticRoleRow`, `ModuleSemanticRoleRow` → `config/datasets/semantic_roles.py`
4. **Deleted all adapter classes** (12 classes, ~3,100 LOC)
5. **Deleted all adapter tests** (~3,560 LOC)
6. **Created migration stub** with helpful `ImportError` messages

### Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Source Files | 11 | 1 | -10 |
| Source LOC | ~3,100 | ~65 | -3,035 |
| Test Files | 10 | 0 | -10 |
| Test LOC | ~3,560 | 0 | -3,560 |
| **Total** | ~6,660 | ~65 | **-6,595** |

---

## Part 2: Graphs Adapters

### Current State

```
src/codeintel/graphs/adapters/
├── __init__.py               (50 LOC)  - Package exports
├── duckdb_storage.py         (224 LOC) - DuckDB storage adapter
├── callgraph_persistence.py  (212 LOC) - Call graph edge utilities
├── libcst_parsing.py         (365 LOC) - LibCST parsing adapter
└── nx_engine_adapter.py      (136 LOC) - NetworkX engine wrapper
```

### Updated Usage Analysis (Post-Analytics Migration)

**Critical Finding:** After auditing actual imports, only `callgraph_persistence.py` is actively used.

| Component | Actual Usage | Recommendation |
|-----------|--------------|----------------|
| `DuckDBStorageAdapter` | **0 callers** (exported but unused) | **Delete immediately** |
| `LibCSTParsingAdapter` | **0 callers** (exported but unused) | **Delete immediately** |
| `NxEngineAdapter` | **0 callers** (exported but unused) | **Delete immediately** |
| `dedupe_edge_rows()` | 2 callers: `callgraph.py`, `collection.py` | **Keep** - Active utility |
| `default_edge_key()` | Used by `dedupe_edge_rows()` | **Keep** - Internal helper |
| `persist_call_graph_edges()` | **0 callers** | **Delete** - Unused |

**Verified Callers:**
```
src/codeintel/build/plugins/graphs/builders/callgraph.py:45
    from codeintel.graphs.adapters.callgraph_persistence import dedupe_edge_rows

src/codeintel/graphs/compute/callgraph/collection.py:16
    from codeintel.graphs.adapters.callgraph_persistence import (
        dedupe_edge_rows,
        default_edge_key,
    )
```

### Revised Migration Strategy

#### Phase 2.1: Delete Unused Adapter Classes (Immediate)

These adapter classes have **zero usage** and can be deleted immediately:

| File | LOC | Exports | Action |
|------|-----|---------|--------|
| `duckdb_storage.py` | 224 | `DuckDBStorageAdapter` | **Delete** - No callers |
| `libcst_parsing.py` | 365 | `LibCSTParsingAdapter` | **Delete** - No callers |
| `nx_engine_adapter.py` | 136 | `NxEngineAdapter` | **Delete** - No callers |

**Risk:** None - grep confirms zero imports outside `__init__.py`

#### Phase 2.2: Consolidate Callgraph Utilities

Move the actively used utilities to a more appropriate location:

**Option A (Recommended):** Rename in place
```
src/codeintel/graphs/adapters/callgraph_persistence.py
    → src/codeintel/graphs/compute/callgraph/persistence.py
```

**Option B:** Keep in adapters (lower risk)
- Keep `callgraph_persistence.py` as-is
- Delete the unused adapter files
- Update `__init__.py` to only export utility functions

**Actions:**
1. Delete `duckdb_storage.py`, `libcst_parsing.py`, `nx_engine_adapter.py`
2. Either move or keep `callgraph_persistence.py`
3. Update `__init__.py` exports
4. Update 2 import statements if moving

#### Phase 2.3: Clean Up Unused Utilities

Within `callgraph_persistence.py`, these functions appear unused:
- `persist_call_graph_edges()`
- `persist_call_graph_nodes()`

**Actions:**
1. Grep verify no callers for persist functions
2. If confirmed unused, delete them
3. Keep only `dedupe_edge_rows()` and `default_edge_key()`

### Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Files | 5 | 1-2 |
| LOC | ~950 | ~100-150 |
| Adapter Classes | 3 | 0 |
| Utility Functions | 5 | 2 |

**Estimated Work:** ~30 minutes (mostly deletions)

---

## Part 3: Ingestion Adapters

### Current State

```
src/codeintel/ingestion/adapters/
├── __init__.py               (30 LOC)   - Package exports
├── duckdb_storage.py         (195 LOC)  - DuckDB storage adapter
├── filesystem_discovery.py   (233 LOC)  - Module discovery adapter
├── hash_change_detection.py  (310 LOC)  - Change detection adapter
├── tool_runner.py            (527 LOC)  - Tool execution adapter
└── build_tool_adapter.py     (307 LOC)  - Build protocol bridge
```

### Updated Usage Analysis (Post-Analytics Migration)

**Critical Finding:** Ingestion adapters are **heavily used** by Hamilton build plugins and cannot be simply deleted.

**Verified Callers:**

| Component | Callers | Caller Locations |
|-----------|---------|------------------|
| `DuckDBStorageAdapter` | 10 | `docstrings_plugin.py`, `typing_plugin.py`, `coverage_plugin.py`, `scip_plugin.py`, `repo_scan.py`, `config_plugin.py`, `cst_extract.py`, `tests_plugin.py`, `ast_extract.py`, `tracker.py`, `compute/__init__.py` |
| `FilesystemDiscoveryAdapter` | 5 | `docstrings_plugin.py`, `typing_plugin.py`, `repo_scan.py`, `config_plugin.py`, `analytics/entrypoints/core.py`, `compute/__init__.py` |
| `BuildToolAdapter` | 2 | `coverage_plugin.py`, `scip_plugin.py` |
| `HashChangeDetectionAdapter` | 1 | `tracker.py` |
| `ToolRunnerAdapter` | 0 | Not directly imported (used internally) |

### Revised Migration Strategy

**⚠️ Key Insight:** Unlike analytics adapters, ingestion adapters provide **real value**:
- They implement well-defined port protocols
- They're used by 9+ active Hamilton plugins
- Removing them would require extensive plugin refactoring

#### Phase 3.1: Keep Core Adapters (No Changes)

These adapters are actively used and should be **kept**:

| Adapter | Usage Count | Recommendation |
|---------|-------------|----------------|
| `DuckDBStorageAdapter` | 10+ callers | **Keep** - Implements `IngestStoragePort` |
| `FilesystemDiscoveryAdapter` | 5 callers | **Keep** - Pure static utilities |
| `BuildToolAdapter` | 2 callers | **Keep** - Protocol bridge pattern |
| `ToolRunnerAdapter` | internal | **Keep** - Tool abstraction layer |

**Rationale:**
- These adapters follow the hexagonal architecture pattern correctly
- They decouple plugins from storage implementation details
- Removing them would couple plugins directly to `StorageGateway`

#### Phase 3.2: Evaluate HashChangeDetectionAdapter (Optional)

`HashChangeDetectionAdapter` has only 1 caller (`tracker.py`):

```python
class HashChangeDetectionAdapter:
    def compute_changes(self, request, current_modules) -> ChangeSet
    def load_previous_state(self, repo, language) -> Mapping[str, FileDigest]
    def persist_current_state(self, repo, language, digests, commit) -> int
```

**Options:**
1. **Keep as-is** (recommended) - Low complexity, single user, working code
2. **Inline into ChangeTracker** - Only if tracker is rewritten for other reasons

#### Phase 3.3: Documentation Update

Rather than migration, focus on documentation:

1. Add module docstrings explaining the port/adapter pattern
2. Document how adapters relate to Hamilton `ctx.gateway`
3. Clarify when to use adapters vs. direct gateway access

**Example docstring update:**
```python
"""DuckDB storage adapter implementing IngestStoragePort.

This adapter provides a stable interface for ingestion plugins while
allowing the underlying storage implementation to evolve. Plugins
should use this adapter rather than accessing StorageGateway directly.

For new Hamilton plugins, prefer:
- `ctx.write_table()` for simple row writes
- This adapter for complex storage operations

See Also
--------
- codeintel.build.context.TargetExecutionContext
- codeintel.storage.gateway.StorageGateway
"""
```

### Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Files | 6 | 6 (unchanged) |
| LOC | ~1,600 | ~1,600 (unchanged) |
| Adapter Classes | 5 | 5 (unchanged) |

**Conclusion:** Ingestion adapters are well-designed and actively used. No significant changes needed beyond documentation improvements.

---

## Implementation Timeline

### Completed

```
✅ Week 1: Part 1 Complete (Analytics Adapters)
   - Extracted DeleteScope to analytics/utilities/persistence.py
   - Moved row builders to analytics/compute/row_builders/
   - Extracted data types to config/datasets/
   - Deleted all adapter classes (~3,100 LOC)
   - Deleted all adapter tests (~3,560 LOC)
   - Created migration error stub
```

### Remaining Work

```
Week 2: Part 2 (Graphs Adapters) - ~30 minutes estimated
   - Delete unused adapter files (duckdb_storage.py, libcst_parsing.py, nx_engine_adapter.py)
   - Keep or move callgraph_persistence.py utilities
   - Update __init__.py exports
   - Update 2 import statements

Week 3: Part 3 (Ingestion Adapters) - Documentation only
   - No code changes needed
   - Update docstrings to explain port/adapter pattern
   - Document relationship to Hamilton ctx.gateway
```

---

## Risk Assessment

### Completed Risks (Part 1 - Mitigated)

| Risk | Outcome | Mitigation Applied |
|------|---------|-------------------|
| Breaking analytics compute | ✅ No issues | All imports updated, tests pass |
| Missing usage patterns | ✅ No issues | grep audit found all callers |
| Test failures | ✅ No issues | Tests updated/deleted appropriately |

### Remaining Risks (Parts 2-3)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking graphs callgraph utilities | Low | Low | Only 2 callers, grep verified |
| Breaking ingestion plugins | N/A | N/A | No changes planned (documentation only) |

---

## Verification Steps

### Before Each Phase

```bash
# 1. Check all usages of target adapter/module
grep -rn "from codeintel.{domain}.adapters" src/ tests/

# 2. Run affected tests
uv run pytest tests/{domain}/ -q

# 3. Run type checking
uv run pyright src/codeintel/{domain}/adapters/
uv run pyrefly check
```

### After Each Phase

```bash
# 1. Full quality check
uv run python -m tools.quality_report

# 2. Full test suite
uv run pytest -q

# 3. Verify no runtime import errors
python -c "from codeintel.{domain} import *"
```

---

## Success Criteria

### Completed (Part 1)

| Criteria | Target | Actual | Status |
|----------|--------|--------|--------|
| Analytics adapters LOC | ~500 | ~65 (stub only) | ✅ Exceeded |
| Analytics tests pass | All | All | ✅ Complete |
| Migration guidance | Clear errors | ImportError with instructions | ✅ Complete |
| Data types preserved | In new locations | config/datasets/, compute/functions/ | ✅ Complete |

### Remaining (Parts 2-3)

| Criteria | Target | Status |
|----------|--------|--------|
| Graphs adapters LOC | ~100-150 | Pending |
| Ingestion adapters | Documentation only | Pending |
| All tests pass | After each phase | Pending |

---

## Appendix A: File Inventory

### Analytics Adapters - COMPLETED

**Before:** ~3,100 LOC across 11 files
**After:** ~65 LOC (error stub only)

| File | Status | New Location |
|------|--------|--------------|
| `base.py` | ✅ Deleted | `DeleteScope` → `utilities/persistence.py` |
| `profiles.py` | ✅ Deleted | N/A (unused) |
| `functions.py` | ✅ Deleted | `FunctionGoid` → `compute/functions/goids.py` |
| `subsystems.py` | ✅ Deleted | N/A (unused) |
| `semantic_roles.py` | ✅ Deleted | Types → `config/datasets/semantic_roles.py` |
| `entrypoints.py` | ✅ Deleted | N/A (unused) |
| `data_models.py` | ✅ Deleted | N/A (unused) |
| `dependencies.py` | ✅ Deleted | Types → `config/datasets/dependencies.py` |
| `schema_adapter.py` | ✅ Deleted | N/A (use `ctx.write_validated_table()`) |
| `graphs/` | ✅ Deleted | Moved to `compute/row_builders/` |

### Graphs Adapters (~950 LOC) - PENDING

| File | LOC | Actual Usage | Recommended Action |
|------|-----|--------------|-------------------|
| `duckdb_storage.py` | 224 | 0 callers | **Delete** |
| `callgraph_persistence.py` | 212 | 2 callers | **Keep** (move to `compute/callgraph/`) |
| `libcst_parsing.py` | 365 | 0 callers | **Delete** |
| `nx_engine_adapter.py` | 136 | 0 callers | **Delete** |

### Ingestion Adapters (~1,600 LOC) - NO CHANGES

| File | LOC | Actual Usage | Recommended Action |
|------|-----|--------------|-------------------|
| `duckdb_storage.py` | 195 | 10+ callers | **Keep** |
| `filesystem_discovery.py` | 233 | 5 callers | **Keep** |
| `hash_change_detection.py` | 310 | 1 caller | **Keep** |
| `tool_runner.py` | 527 | Internal | **Keep** |
| `build_tool_adapter.py` | 307 | 2 callers | **Keep** |

---

## Appendix B: Hamilton Context API Reference

The Hamilton build context (`TargetExecutionContext`) provides these methods as replacements for adapter patterns:

```python
class TargetExecutionContext:
    # Storage access
    @property
    def gateway(self) -> StorageGateway: ...
    
    # Row writing (replaces adapters)
    def write_table(
        self,
        table_key: str,
        rows: Sequence[tuple | dict],
        *,
        validate: bool = True,
    ) -> int: ...
    
    # DataFrame writing with Pandera validation
    def write_validated_table(
        self,
        table_key: str,
        df: pd.DataFrame,
        *,
        strict: bool = True,
    ) -> int: ...
    
    # Resources (tools, providers)
    @property
    def resources(self) -> ContextResources: ...
```

The `StorageGateway` provides:

```python
class StorageGateway:
    @property
    def policy(self) -> DuckDBPolicyBackend: ...
    
    @property
    def ibis(self) -> IbisGateway: ...
    
    def execute(self, sql: str, params: list | None = None) -> DuckDBConnection: ...
```

---

## Appendix C: Related Documentation

- [Legacy Decommissioning Summary](./Legacy_Decommissioning_Summary.md)
- [Legacy Decommissioning Plan Phase 2](./Legacy_Decommissioning_Plan_Phase2.md)
- [Hamilton Phase 4 Design](./Hamilton_apache_phase4.md)

