# Adapters Migration Plan: Transition to Hamilton-Native Patterns

**Document Version:** 1.0  
**Created:** 2025-12-13  
**Status:** Proposed  
**Scope:** `analytics/adapters/`, `graphs/adapters/`, `ingestion/adapters/`

---

## Executive Summary

This document outlines a phased migration plan for transitioning adapter implementations across three domain packages to Hamilton-native patterns. The adapters were designed for the legacy plugin orchestration system and represent ~4,800+ LOC that can be consolidated, simplified, or eliminated as the Hamilton build system matures.

### Package Overview

| Package | Files | LOC | Hamilton Plugin Usage | Migration Priority |
|---------|-------|-----|----------------------|-------------------|
| `ingestion/adapters/` | 6 | ~1,800 | **Heavy** (11+ plugins) | Phase 3 (Last) |
| `graphs/adapters/` | 5 | ~950 | **Light** (1 plugin) | Phase 2 |
| `analytics/adapters/` | 11 | ~2,100 | **Minimal** (1 import) | Phase 1 (First) |

---

## Part 1: Analytics Adapters

### Current State

```
src/codeintel/analytics/adapters/
├── __init__.py           (85 LOC)  - Package exports
├── base.py               (397 LOC) - Base adapter classes
├── profiles.py           (345 LOC) - Profile table adapters
├── functions.py          (505 LOC) - Function metrics adapters
├── subsystems.py         (249 LOC) - Subsystem classification adapters
├── semantic_roles.py     (460 LOC) - Semantic role adapters
├── entrypoints.py        (188 LOC) - Entrypoint detection adapters
├── data_models.py        (108 LOC) - Data model usage adapters
├── dependencies.py       (354 LOC) - Dependency adapters
├── schema_adapter.py     (212 LOC) - Schema validation mixin
└── graphs/
    ├── __init__.py       - Graph metric exports
    ├── graph_metrics.py  - Function graph metric row builders
    ├── graph_metrics_ext.py - Extended graph metrics
    ├── subsystem_graph_metrics.py - Subsystem graph metrics
    └── symbol_graph_metrics.py - Symbol graph metrics
```

### Usage Analysis

| Component | Used By | Hamilton Equivalent |
|-----------|---------|---------------------|
| `DeleteScope` (base.py) | 1 build plugin, 5 analytics compute modules | Move to `analytics/utilities/` |
| `BatchAdapter`, `AnalyticsAdapter` | 6 concrete adapters (not build plugins) | `ctx.write_table()` |
| `SchemaValidationMixin` | 3 profile adapters | `ctx.write_validated_table()` |
| `graphs/*.py` row builders | 5 analytics compute modules | **Keep** - These are pure functions |

### Migration Strategy

#### Phase 1.1: Extract Common Utilities (Week 1)

**Move `DeleteScope` to utilities:**

```python
# FROM: analytics/adapters/base.py
# TO:   analytics/utilities/persistence.py

@dataclass(frozen=True)
class DeleteScope:
    """Specification for scoped deletion before insert."""
    repo: str
    commit: str
    columns: tuple[str, ...] | None = None
```

**Actions:**
1. Create `analytics/utilities/persistence.py`
2. Move `DeleteScope` class
3. Update imports in:
   - `build/plugins/analytics/functions/ast_features.py`
   - `analytics/graphs/graph_metrics.py`
   - `analytics/graphs/graph_metrics_ext.py`
   - `analytics/graphs/module_graph_metrics_ext.py`
   - `analytics/utilities/datasets.py`

#### Phase 1.2: Rename Graph Row Builders (Week 1)

The `analytics/adapters/graphs/` directory contains **row builder functions**, not adapters:

```python
# These are pure functions, not adapter classes:
- build_function_graph_metric_rows()
- build_module_graph_metric_rows()
- persist_graph_metrics()
```

**Actions:**
1. Rename `analytics/adapters/graphs/` → `analytics/compute/graph_row_builders/`
2. Update all imports in `analytics/graphs/*.py`
3. Export from `analytics/compute/__init__.py`

#### Phase 1.3: Deprecate Unused Adapter Classes (Week 2)

These adapter classes have **zero usage** in Hamilton plugins:

| Adapter | Status | Action |
|---------|--------|--------|
| `FunctionProfileAdapter` | Unused | Add deprecation warning |
| `FileProfileAdapter` | Unused | Add deprecation warning |
| `ModuleProfileAdapter` | Unused | Add deprecation warning |
| `FunctionMetricsAdapter` | Unused | Add deprecation warning |
| `FunctionTypesAdapter` | Unused | Add deprecation warning |
| `SubsystemsAdapter` | Unused | Add deprecation warning |
| `SubsystemModulesAdapter` | Unused | Add deprecation warning |
| `SemanticRolesFunctionsAdapter` | Unused | Add deprecation warning |
| `SemanticRolesModulesAdapter` | Unused | Add deprecation warning |
| `EntrypointsAdapter` | Unused | Add deprecation warning |
| `EntrypointTestsAdapter` | Unused | Add deprecation warning |
| `DataModelUsageAdapter` | Unused | Add deprecation warning |

**Deprecation Pattern:**

```python
import warnings

class FunctionProfileAdapter(BatchAdapter[dict[str, Any]], SchemaValidationMixin):
    """Adapter for analytics.function_profile table.
    
    .. deprecated:: 4.0.0
        Use ``ctx.write_table()`` or ``ctx.write_validated_table()`` instead.
        This adapter will be removed in version 5.0.0.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "FunctionProfileAdapter is deprecated. "
            "Use ctx.write_table() or ctx.write_validated_table() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
```

#### Phase 1.4: Delete Deprecated Adapters (Week 4+)

After deprecation period (1-2 release cycles):

1. Delete adapter class files
2. Remove from `__init__.py` exports
3. Delete empty `adapters/` directory
4. Update any remaining internal references

### Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Files | 11 | 0 (or 1 for graphs row builders) |
| LOC | ~2,100 | ~500 (row builders only) |
| Adapter Classes | 12 | 0 |

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

### Usage Analysis

| Component | Used By | Hamilton Equivalent |
|-----------|---------|---------------------|
| `DuckDBStorageAdapter` | `graphs/resources/storage.py` | `build/context.py` + policy backend |
| `persist_call_graph_edges()` | 1 build plugin | **Keep** - utility function |
| `dedupe_edge_rows()` | 1 build plugin | **Keep** - utility function |
| `LibCSTParsingAdapter` | `graphs/resources/graphs.py` | Direct LibCST usage |
| `NxEngineAdapter` | `graphs/resources/graphs.py` | Direct engine usage |

### Migration Strategy

#### Phase 2.1: Keep Persistence Utilities (Week 2)

The `callgraph_persistence.py` contains **pure utility functions** that are actively used:

```python
# These stay - they're utility functions, not adapters:
- dedupe_edge_rows()      # Used by callgraph.py plugin
- default_edge_key()      # Helper for deduplication
- persist_call_graph_edges()  # Useful batch operation
- persist_call_graph_nodes()  # Useful batch operation
```

**Actions:**
1. Rename file to `graphs/compute/callgraph_utils.py` or keep in adapters
2. Consider moving to `graphs/persistence/` for clarity

#### Phase 2.2: Evaluate DuckDBStorageAdapter (Week 2-3)

The `DuckDBStorageAdapter` duplicates functionality available via:
- `StorageGateway.policy` (DuckDBPolicyBackend)
- `ctx.gateway` in build context

**Assessment Questions:**
1. Is `DuckDBStorageAdapter` used directly by any compute modules?
2. Can `graphs/resources/storage.py` use `ctx.gateway` directly?

**Actions:**
1. Audit all usages of `graphs/adapters/duckdb_storage.py`
2. Migrate callers to use `gateway.policy` directly
3. Mark as deprecated once no callers remain

#### Phase 2.3: Evaluate Parsing/Engine Adapters (Week 3)

`LibCSTParsingAdapter` and `NxEngineAdapter` wrap external libraries:

| Adapter | Purpose | Recommendation |
|---------|---------|----------------|
| `LibCSTParsingAdapter` | Wraps LibCST for module parsing | **Evaluate** - May be useful abstraction |
| `NxEngineAdapter` | Wraps NxGraphEngine | **Deprecate** - Thin wrapper adds no value |

**Actions for LibCSTParsingAdapter:**
1. Assess if the abstraction provides testability benefits
2. If primarily for testing, keep but rename to `graphs/parsing/libcst_adapter.py`
3. If just a thin wrapper, inline into callers

**Actions for NxEngineAdapter:**
1. Mark as deprecated immediately (adds no value)
2. Update `graphs/resources/graphs.py` to use `NxGraphEngine` directly
3. Delete after one release cycle

### Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Files | 5 | 2-3 |
| LOC | ~950 | ~400-600 |
| Adapter Classes | 3 | 0-1 |

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

### Usage Analysis

**Critical Finding:** Ingestion adapters are **heavily used** by Hamilton build plugins.

| Component | Hamilton Plugin Usage | Direct Callers |
|-----------|----------------------|----------------|
| `DuckDBStorageAdapter` | 11 plugins | `tracker.py` |
| `FilesystemDiscoveryAdapter` | 3 plugins | - |
| `BuildToolAdapter` | 2 plugins | - |
| `HashChangeDetectionAdapter` | 0 plugins | `tracker.py` |
| `ToolRunnerAdapter` | 0 plugins | internal |

### Migration Strategy

**⚠️ Important:** Ingestion adapters require careful migration due to heavy usage.

#### Phase 3.1: Consolidate Storage Access (Week 4-5)

The `DuckDBStorageAdapter` implements `IngestStoragePort`:

```python
class DuckDBStorageAdapter(IngestStoragePort):
    def write_batch(self, table_key, rows, *, scope=None) -> BatchResult
    def delete_by_params(self, table_key, params) -> int
    def delete_by_paths(self, table_key, paths, **kwargs) -> int
    def execute_query(self, sql, params=None) -> QueryResult
    def fetch_dataframe(self, sql, params=None) -> pd.DataFrame
```

**Hamilton Equivalents:**

| Adapter Method | Hamilton Equivalent |
|----------------|---------------------|
| `write_batch()` | `ctx.write_table()` + `gateway.policy.bulk_insert()` |
| `delete_by_params()` | `gateway.policy.delete_for_snapshot()` |
| `delete_by_paths()` | `gateway.ibis.delete()` with filter |
| `execute_query()` | `gateway.execute()` |
| `fetch_dataframe()` | `gateway.execute().fetch_df()` |

**Recommended Approach:**
1. **Don't delete** - Keep the adapter as a convenience layer
2. **Simplify** - Make it a thin wrapper around `gateway.policy`
3. **Document** - Mark as "compatibility layer" in docstrings

#### Phase 3.2: Evaluate Discovery Adapter (Week 5)

`FilesystemDiscoveryAdapter` provides module discovery utilities:

```python
class FilesystemDiscoveryAdapter:
    @staticmethod
    def discover_modules(repo_root, profile) -> Sequence[ModuleRecord]
    @staticmethod
    def iter_modules(module_map, repo_root, **kwargs) -> Iterator[ModuleRecord]
    @staticmethod
    def read_module_source(record) -> str | None
```

**Assessment:**
- These are **static utility methods**, not adapter state
- Could be moved to `ingestion/compute/discovery.py`

**Actions:**
1. Keep adapter for now (low migration risk)
2. Consider renaming to `ingestion/discovery/filesystem.py`
3. No deprecation needed - pure functions are fine

#### Phase 3.3: Keep Tool Adapters (Week 5-6)

`BuildToolAdapter` and `ToolRunnerAdapter` bridge different tool abstractions:

| Adapter | Purpose | Recommendation |
|---------|---------|----------------|
| `BuildToolAdapter` | Bridge build protocols → ingestion ports | **Keep** - Active bridge pattern |
| `ToolRunnerAdapter` | Wrap ToolService for port compliance | **Keep** - Useful abstraction |

**Rationale:**
- These adapters provide **real value** by normalizing result types
- The build system and ingestion system have different type hierarchies
- Keeping the bridge is cleaner than forcing one system to use another's types

#### Phase 3.4: Migrate HashChangeDetectionAdapter (Week 6)

`HashChangeDetectionAdapter` is used only by `tracker.py`:

```python
class HashChangeDetectionAdapter:
    def compute_changes(self, request, current_modules) -> ChangeSet
    def load_previous_state(self, repo, language) -> Mapping[str, FileDigest]
    def persist_current_state(self, repo, language, digests, commit) -> int
```

**Assessment:**
- Could be inlined into `ChangeTracker`
- Or kept as-is (low complexity, single user)

**Actions:**
1. Low priority - keep as-is unless refactoring tracker
2. Consider inlining if tracker is rewritten

### Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Files | 6 | 4-5 |
| LOC | ~1,600 | ~1,200-1,400 |
| Adapter Classes | 5 | 3-4 |

---

## Implementation Timeline

```
Week 1: Phase 1.1-1.2 (Analytics utilities extraction)
        - Move DeleteScope to analytics/utilities/persistence.py
        - Rename analytics/adapters/graphs/ to analytics/compute/graph_row_builders/
        
Week 2: Phase 1.3 + Phase 2.1 (Deprecation warnings + graphs utilities)
        - Add deprecation warnings to unused analytics adapters
        - Evaluate graphs persistence utilities
        
Week 3: Phase 2.2-2.3 (Graphs adapter evaluation)
        - Migrate DuckDBStorageAdapter usages
        - Deprecate NxEngineAdapter
        
Week 4-5: Phase 3.1-3.2 (Ingestion storage consolidation)
        - Simplify DuckDBStorageAdapter
        - Evaluate FilesystemDiscoveryAdapter
        
Week 6: Phase 3.3-3.4 (Final cleanup)
        - Document tool adapter architecture decision
        - Evaluate HashChangeDetectionAdapter
        
Week 8+: Deletion phase (after deprecation period)
        - Delete deprecated analytics adapters
        - Delete deprecated graphs adapters
        - Clean up empty directories
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking internal analytics compute | Medium | High | Run full test suite before each phase |
| Breaking ingestion plugins | Low | Critical | Phased migration with compatibility shims |
| Missing usage patterns | Medium | Medium | grep/ripgrep audit before each deletion |
| Test failures | Medium | Low | Update tests alongside code changes |

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

1. **Analytics adapters:** Reduce from 2,100 LOC to ~500 LOC (row builders only)
2. **Graphs adapters:** Reduce from 950 LOC to ~400-600 LOC
3. **Ingestion adapters:** Simplify and document; minimal LOC reduction expected
4. **All tests pass** after each phase
5. **No deprecation warnings** in Hamilton plugin code paths
6. **Clear documentation** of remaining adapter purposes

---

## Appendix A: File Inventory

### Analytics Adapters (~2,100 LOC)

| File | LOC | Exports | Action |
|------|-----|---------|--------|
| `base.py` | 397 | `DeleteScope`, `BatchAdapter`, etc. | Extract `DeleteScope`, deprecate rest |
| `profiles.py` | 345 | 3 profile adapters | Deprecate |
| `functions.py` | 505 | 2 function adapters | Deprecate |
| `subsystems.py` | 249 | 2 subsystem adapters | Deprecate |
| `semantic_roles.py` | 460 | 2 semantic role adapters | Deprecate |
| `entrypoints.py` | 188 | 2 entrypoint adapters | Deprecate |
| `data_models.py` | 108 | 1 data model adapter | Deprecate |
| `dependencies.py` | 354 | 1 dependency adapter | Deprecate |
| `schema_adapter.py` | 212 | `SchemaValidationMixin` | Deprecate (use ctx method) |
| `graphs/__init__.py` | ~20 | Exports | Move to compute/ |
| `graphs/*.py` | ~500 | Row builder functions | Keep, rename |

### Graphs Adapters (~950 LOC)

| File | LOC | Exports | Action |
|------|-----|---------|--------|
| `duckdb_storage.py` | 224 | `DuckDBStorageAdapter` | Evaluate, possibly deprecate |
| `callgraph_persistence.py` | 212 | Persistence utilities | Keep |
| `libcst_parsing.py` | 365 | `LibCSTParsingAdapter` | Evaluate |
| `nx_engine_adapter.py` | 136 | `NxEngineAdapter` | Deprecate |

### Ingestion Adapters (~1,600 LOC)

| File | LOC | Exports | Action |
|------|-----|---------|--------|
| `duckdb_storage.py` | 195 | `DuckDBStorageAdapter` | Simplify, keep |
| `filesystem_discovery.py` | 233 | `FilesystemDiscoveryAdapter` | Keep |
| `hash_change_detection.py` | 310 | `HashChangeDetectionAdapter` | Evaluate |
| `tool_runner.py` | 527 | `ToolRunnerAdapter` | Keep |
| `build_tool_adapter.py` | 307 | `BuildToolAdapter` | Keep |

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

