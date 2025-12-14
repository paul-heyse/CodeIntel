# Build System Refinement Plan

## Overview

This document outlines opportunities to consolidate shared functionality and refine the build system towards a best-in-class implementation.

---

## Completed Work

### Part 1: Plugin Migration to MetadataPlugin ✅ COMPLETE

**28 plugins migrated** to use `MetadataPlugin`, eliminating ~700 lines of boilerplate:

| Category | Plugins Migrated | Pattern |
|----------|-----------------|---------|
| Category A (Simple) | 11 | Direct migration, no custom `__init__` |
| Category B (With Options) | 12 | Removed custom `resolve_options()` |
| Category C (With Factories) | 4 | Migrated to `FactoryPlugin` base |
| Category D (Legacy) | 2 | Added `_core_metadata` (core.py, validation.py) |

**New pattern:**
```python
class MyPlugin(MetadataPlugin):
    _core_metadata: ClassVar[CorePluginMetadata] = MY_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        opts = self.resolve_options(MyOptionsType)  # Generic inherited method
        ...
```

### Part 2: Row Count Computation ✅ COMPLETE

Created shared helper in `build/plugins/_helpers.py`:
- `compute_row_counts(ctx)` - Computes row counts for output tables with snapshot filtering
- Already used by `hotspots/build.py`, `scip_plugin.py`, `repo_scan.py`

### Part 3: Options Resolver Pattern ✅ COMPLETE

Added generic `resolve_options()` to `MetadataPlugin`:
- Uses `TypeVar` for type-safe return values
- Automatically uses `_core_metadata.options_model` when type not specified
- Falls back to default options if no resolver configured

### Part 4: Factory Plugin Pattern ✅ COMPLETE

Created `FactoryPlugin` base class in `build/plugin.py`:
- Consolidates factory pattern for ingestion plugins
- Handles `storage_factory`, `discovery_factory`, `step_factory`
- Migrated: `typing_plugin.py`, `docstrings_plugin.py`, `cst_extract.py`, `ast_extract.py`

### Part 5: Module Path Helper Consolidation ✅ COMPLETE

Removed duplicated `_paths_to_modules` and `_get_module_paths` from:
- `ingestion/typing_plugin.py`
- `ingestion/docstrings_plugin.py`
- `ingestion/cst_extract.py`
- `ingestion/ast_extract.py`

Now all use shared `get_module_paths` and `paths_to_modules` from `ingestion/helpers.py`.

### Part 6: Legacy Plugin Metadata ✅ COMPLETE

Migrated remaining plugins to `MetadataPlugin` pattern:
- `graphs/metrics/core.py` - `CoreMetricsPlugin`
- `graphs/validation.py` - `GraphValidationPlugin`

### Part 10: Error Handling ✅ COMPLETE (partial)

Replaced generic exceptions with structured error types in key locations:
- `unified_registry.py` → `RegistryValidationError`
- `hamilton/native/executor.py` → `TargetNotFoundError`
- `targets.py` → `CycleDetectedError`
- Ingestion plugins → `GatewayNotAvailableError`

### Part 12: Source Root Helper Consolidation ✅ COMPLETE

Created `get_source_root()` in `build/plugins/_helpers.py`:

```python
def get_source_root(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    *,
    fallback: Path | None = None,
) -> Path:
    """Retrieve source root from core.snapshots with fallback."""
```

Updated 4 graph builders:
- `graphs/builders/callgraph.py`
- `graphs/builders/import_graph.py`
- `graphs/builders/cfg_dfg.py`
- `graphs/builders/goid.py`

**Lines removed:** ~80

### Part 13: Test Path Detection Consolidation ✅ COMPLETE

Created `is_test_path()` in `build/plugins/_helpers.py`:

```python
def is_test_path(path: str) -> bool:
    """Check whether a path appears to be a test file."""
```

Updated 4 files:
- `graphs/builders/symbol_uses.py`
- `graphs/builders/cfg_dfg.py`
- `graphs/builders/goid.py`
- `ingestion/helpers.py`

**Lines removed:** ~40

### Part 14: Path Filtering Consolidation ✅ COMPLETE

Created `filter_paths()` in `build/plugins/_helpers.py`:

```python
def filter_paths(
    paths: Iterable[str],
    *,
    scope_paths: list[str] | None = None,
    include_tests: bool = True,
) -> list[str]:
    """Filter paths by scope and test inclusion."""
```

Updated 4 files:
- `graphs/builders/callgraph.py` - removed `_filter_paths_by_scope`
- `graphs/builders/cfg_dfg.py` - removed `_filter_paths`
- `graphs/builders/goid.py` - removed `_filter_tracked_files`
- `ingestion/scip_plugin.py` - removed `_filter_paths`

**Lines removed:** ~50

---

## Remaining Work & New Opportunities

### Part 15: Dict-Based Filtering Consolidation (NEW - MEDIUM PRIORITY)

#### Problem

Two plugins have dict-based filtering that can't use `filter_paths()` directly:

```python
# In import_graph.py:
def _filter_paths_by_scope(
    module_by_path: Mapping[str, str],
    scope_paths: list[str] | None,
) -> dict[str, str]:
    if not scope_paths:
        return dict(module_by_path)
    prefixes = tuple(scope_paths)
    return {path: module for path, module in module_by_path.items() if path.startswith(prefixes)}

# In symbol_uses.py (3 functions with identical pattern):
def _filter_module_map(module_map: dict[str, str], options: SymbolUsesOptions) -> dict[str, str]:
    return {
        path: module
        for path, module in module_map.items()
        if _matches_scope(path, options.scope_paths)
        and (options.include_tests or not is_test_path(path))
    }
```

**Files with this pattern:**
- `graphs/builders/import_graph.py` - `_filter_paths_by_scope`
- `graphs/builders/symbol_uses.py` - `_matches_scope`, `_filter_module_map`, `_filter_path_to_goid_map`, `_filter_occurrences`

#### Solution

Add `filter_mapping()` helper to `build/plugins/_helpers.py`:

```python
def filter_mapping(
    mapping: Mapping[str, T],
    *,
    scope_paths: list[str] | None = None,
    include_tests: bool = True,
) -> dict[str, T]:
    """Filter a path-keyed mapping by scope and test inclusion.

    Parameters
    ----------
    mapping
        Mapping with relative paths as keys.
    scope_paths
        Optional path prefixes to include.
    include_tests
        Whether to include test paths.

    Returns
    -------
    dict[str, T]
        Filtered mapping.
    """
    result = dict(mapping)

    if scope_paths:
        prefixes = tuple(scope_paths)
        result = {k: v for k, v in result.items() if k.startswith(prefixes)}

    if not include_tests:
        result = {k: v for k, v in result.items() if not is_test_path(k)}

    return result
```

Also remove `_matches_scope` from symbol_uses.py (can be inlined or use scope check directly).

**Estimated savings:** ~40 lines

---

### Part 16: Persistence Pattern Consolidation (MEDIUM PRIORITY - REASSESSED)

#### Problem

10 persistence functions across 5 graph builder files follow an identical 3-step pattern:

```python
def _persist_X(gateway, rows, repo, commit) -> int:
    if not rows:
        return 0
    gateway.policy.ensure_table("table.key")
    gateway.policy.delete_for_snapshot("table.key", repo=repo, commit=commit)
    gateway.policy.bulk_insert("table.key", [row.to_tuple() for row in rows])
    return len(rows)
```

**Files with this pattern (10 functions total):**
- `callgraph.py` - `_persist_nodes`, `_persist_edges` (edge has JSON serialization)
- `import_graph.py` - `_persist_import_modules`, `_persist_import_edges`
- `cfg_dfg.py` - `_persist_cfg_blocks`, `_persist_cfg_edges`, `_persist_dfg_edges`
- `goid.py` - `_persist_goid_rows`, `_persist_crosswalk_rows`
- `symbol_uses.py` - persistence inline (symbol use edges)

#### Complexity Note

The `_persist_edges` in callgraph.py has custom JSON serialization logic for `evidence_json`, making it unsuitable for a generic helper without significant complexity.

#### Solution

Create a simple `persist_rows()` helper for the 9 standard cases:

```python
def persist_rows(
    gateway: StorageGateway,
    table_key: str,
    rows: Sequence[Any],
    *,
    repo: str,
    commit: str,
) -> int:
    """Persist rows to a table with snapshot cleanup.

    Standard persistence pattern:
    1. Return 0 for empty input
    2. Ensure table exists
    3. Delete existing snapshot data
    4. Bulk insert new rows

    Parameters
    ----------
    gateway
        Storage gateway.
    table_key
        Fully-qualified table name (e.g., "graph.cfg_blocks").
    rows
        Rows to persist. Must have `.to_tuple()` method.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    int
        Number of rows persisted.
    """
    if not rows:
        return 0
    gateway.policy.ensure_table(table_key)
    gateway.policy.delete_for_snapshot(table_key, repo=repo, commit=commit)
    gateway.policy.bulk_insert(table_key, [row.to_tuple() for row in rows])
    return len(rows)
```

Keep `_persist_edges` in callgraph.py for the JSON serialization case.

**Estimated savings:** ~160 lines (9 functions × ~18 lines each)

---

### Part 17: GraphBuilderPlugin Base Class (MEDIUM PRIORITY)

#### Problem

All 5 graph builder plugins share a common `execute()` structure:

```python
async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
    opts = self.resolve_options(XOptions)
    snapshot = ctx.snapshot
    gateway, repo, commit = ctx.gateway, snapshot.repo, snapshot.commit

    try:
        # 1. Load index data
        function_index = load_function_index(gateway, repo=repo, commit=commit)
        
        # 2. Filter paths
        paths = filter_paths(function_index.paths(), scope_paths=opts.scope_paths, ...)

        # 3. Early return if empty
        if not paths:
            log.info("...: No data found, skipping")
            return TargetResult.succeeded(row_counts={...})

        # 4. Get source root
        source_root = snapshot.repo_root or get_source_root(gateway, repo, commit)
        
        # 5. Process and collect results
        # ... plugin-specific logic ...
        
        # 6. Persist and return
        return TargetResult.succeeded(row_counts={...})
    except (RuntimeError, ValueError, OSError) as e:
        return TargetResult.failed(f"... failed: {e}")
```

#### Solution

Create `GraphBuilderPlugin` base class:

```python
class GraphBuilderPlugin(MetadataPlugin, Generic[TOptions]):
    """Base class for graph builder plugins.

    Provides common infrastructure for:
    - Source root resolution with fallback
    - Path filtering by scope and test inclusion
    - Standardized error handling
    - Empty row counts helper

    Subclasses implement `_execute_impl()` with the actual build logic.
    """

    @property
    def empty_row_counts(self) -> dict[str, int]:
        """Return zero row counts for all output tables."""
        return {table: 0 for table in self._core_metadata.produces_tables}

    def get_source_root(self, ctx: TargetExecutionContext) -> Path:
        """Get source root with fallback to snapshot or cwd."""
        if ctx.snapshot.repo_root:
            return ctx.snapshot.repo_root
        return get_source_root(ctx.gateway, ctx.repo, ctx.commit)

    def filter_paths(
        self,
        paths: Iterable[str],
        opts: TOptions,
    ) -> list[str]:
        """Filter paths using options attributes.
        
        Looks for scope_paths, include_tests, include_test_files on opts.
        """
        scope_paths = getattr(opts, "scope_paths", None)
        include_tests = getattr(opts, "include_tests", True)
        include_test_files = getattr(opts, "include_test_files", include_tests)
        return filter_paths(paths, scope_paths=scope_paths, include_tests=include_test_files)

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute with standardized error handling."""
        try:
            return await self._execute_impl(ctx)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"{self.plugin_name} failed: {e}")

    @abstractmethod
    async def _execute_impl(self, ctx: TargetExecutionContext) -> TargetResult:
        """Implement the actual build logic."""
        ...
```

**Files to refactor:**
- `graphs/builders/callgraph.py`
- `graphs/builders/import_graph.py`
- `graphs/builders/cfg_dfg.py`
- `graphs/builders/goid.py`
- `graphs/builders/symbol_uses.py`

**Estimated savings:** ~80 lines + standardized error handling

---

### Part 18: Common Table Loading Patterns (NEW - LOW PRIORITY)

#### Problem

Several graph builders have nearly identical table loading patterns:

```python
# Loading modules (in import_graph.py):
def _load_modules(gateway, repo, commit) -> dict[str, str]:
    modules = gateway.ibis.table("core.modules")
    expr = modules.filter(
        cast("Any", modules.repo == repo) & cast("Any", modules.commit == commit)
    ).select(modules.path, modules.module)
    df = expr.execute()
    return {normalize_rel_path(str(path)): str(module) for ...}

# Loading similar data in goid.py, symbol_uses.py
```

#### Solution

Consider adding common loaders to `_helpers.py`:

```python
def load_module_map(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, str]:
    """Load module name by path mapping from core.modules."""
    ...

def load_path_to_goid_map(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    *,
    kind: str | None = None,
) -> dict[str, int]:
    """Load path to GOID mapping from core.goids."""
    ...
```

This is lower priority as each loader has slight variations.

**Estimated savings:** ~40 lines

---

### Part 19: Ingestion Plugin Step Pattern (NEW - LOW PRIORITY)

#### Problem

Ingestion plugins follow a consistent pattern with storage/tool adapters:

```python
# Repeated in coverage_plugin.py, scip_plugin.py, typing_plugin.py, etc.
storage = DuckDBStorageAdapter(ctx.gateway)
tool = BuildToolAdapter(...)

step = XIngestStep(storage=storage, tools=tool)
result = await step.execute_async(modules, ...)

if not result.success:
    errors = "; ".join(result.errors) if result.errors else "Unknown error"
    return TargetResult.failed(f"X ingest failed: {errors}")

return TargetResult.succeeded(row_counts=result.table_counts or {})
```

The `FactoryPlugin` base class already partially addresses this for storage/discovery factories.

#### Solution

Consider adding an `execute_step()` helper method to `FactoryPlugin`:

```python
async def execute_step(
    self,
    step: Any,
    modules: Sequence[ModuleRecord],
    **kwargs: Any,
) -> TargetResult:
    """Execute an ingestion step with standard error handling."""
    result = await step.execute_async(modules, **kwargs)
    if not result.success:
        errors = "; ".join(result.errors) if result.errors else "Unknown error"
        return TargetResult.failed(f"{self.plugin_name} failed: {errors}")
    return TargetResult.succeeded(row_counts=result.table_counts or {})
```

**Estimated savings:** ~30 lines across 4+ plugins

---

### Part 7: Context Hierarchy Simplification (DEFERRED)

#### Problem

Multiple overlapping context types:
- `MaterializationContext` (deprecated, but still used)
- `ArtifactMaterializationContext` (separate from BuildContext)
- `MaterializationContextProtocol` (protocol for compatibility)

#### Solution

1. Remove `MaterializationContext` entirely (already deprecated)
2. Add artifact support directly to `BuildContext`
3. Audit and remove protocol if no longer needed

---

### Part 8: Native Target Pattern Improvements (DEFERRED)

#### Problem

Native targets still have some repetitive patterns even after `NativeTargetExecutor`:
- Artifact handling requires manual record creation
- Export targets have duplicated JSON/Parquet patterns

#### Solution

##### 8.1 Add artifact support to NativeTargetExecutor

```python
class NativeTargetExecutor:
    def execute_with_artifacts(
        self,
        compute_fn: Callable[[], tuple[dict[str, int], tuple[ArtifactRef, ...]]],
    ) -> TargetRunRecord:
        """Execute with artifact support."""
        ...
```

##### 8.2 Create ExportTargetMixin for common export patterns

```python
class ExportTargetMixin:
    """Shared utilities for export targets."""

    @staticmethod
    def export_to_jsonl(
        data: list[dict[str, Any]],
        output_path: Path,
        *,
        include_metadata: bool = True,
    ) -> tuple[str, int]:
        """Export data to JSONL format."""
        ...

    @staticmethod
    def export_to_parquet(df: pd.DataFrame, output_path: Path) -> bytes:
        """Export DataFrame to Parquet format."""
        ...
```

---

### Part 9: Unused Import Cleanup ✅ COMPLETE

Import cleanup was performed as part of the factory plugin migration.

---

### Part 11: Import Organization (DEFERRED)

#### Problem

Many modules have inconsistent import patterns:
- Some use TYPE_CHECKING guards, some don't
- Lazy imports (`# noqa: PLC0415`) are scattered
- Some circular import workarounds are complex

#### Solution

Establish clear patterns:

1. **Heavy imports** (numpy, pandas, ibis): Always in TYPE_CHECKING
2. **Circular import prevention**: Use lazy imports in `__init__.py`
3. **Protocol types**: Import from canonical facade modules

Create a facade for common type imports in `build/typing.py`.

---

## Implementation Priority

| Priority | Part | Effort | Impact | Status |
|----------|------|--------|--------|--------|
| ✅ | Part 2: Row Count Helper | Low | High | COMPLETE |
| ✅ | Part 1: Plugin Migration | Medium | High | COMPLETE (28/28) |
| ✅ | Part 3: Options Resolver | Low | Medium | COMPLETE |
| ✅ | Part 4: Factory Plugin Base | Medium | Medium | COMPLETE |
| ✅ | Part 5: Module Path Helpers | Low | Medium | COMPLETE |
| ✅ | Part 6: Legacy Plugin Metadata | Low | Medium | COMPLETE |
| ✅ | Part 10: Error Handling | Low | Medium | COMPLETE |
| ✅ | Part 12: Source Root Helper | Low | High | COMPLETE |
| ✅ | Part 13: Test Path Detection | Low | Medium | COMPLETE |
| ✅ | Part 14: Path Filtering | Low | Medium | COMPLETE |
| **1** | **Part 15: Dict Filtering** | **Low** | **Medium** | **Pending** |
| **2** | **Part 16: Persistence Pattern** | **Medium** | **High** | **Pending** |
| **3** | **Part 17: GraphBuilderPlugin** | **Medium** | **High** | **Pending** |
| 4 | Part 18: Table Loaders | Low | Low | Pending |
| 5 | Part 19: Step Execution | Low | Low | Pending |
| 6 | Part 7: Context Simplification | Medium | Medium | Deferred |
| 7 | Part 8: Export Patterns | Low | Low | Deferred |
| 8 | Part 11: Import Organization | Medium | Low | Deferred |

---

## Current State Summary

### Shared Helpers in `_helpers.py`

The `build/plugins/_helpers.py` module now contains:

| Function | Purpose | Consumers |
|----------|---------|-----------|
| `compute_row_counts()` | Count rows for output tables | 3+ plugins |
| `compute_row_count()` | Single table row count | convenience |
| `get_source_root()` | Source root from snapshots | 4 graph builders |
| `is_test_path()` | Test file detection | 4 files |
| `filter_paths()` | Path filtering by scope/tests | 4 files |

### Validation Checklist

After each phase:

```bash
# Lint and format
uv run ruff check --fix src/codeintel/build/
uv run ruff format src/codeintel/build/

# Type checking
uv run pyright src/codeintel/build/
uv run pyrefly check src/codeintel/build/

# Registry validation
uv run python -c "
from codeintel.build.unified_registry import get_unified_registry
reg = get_unified_registry()
print(f'Registry: {len(reg)} targets')
print(f'Native targets: {len(list(reg.native_target_names()))}')
"

# Tests
uv run pytest tests/build/ -q
```

---

## Summary

### Completed (Phase 1 + Phase 2)

- **28 plugins** migrated to `MetadataPlugin` (~700 lines removed)
- **Row count helper** created and used in 3+ plugins
- **Generic `resolve_options()`** added to `MetadataPlugin`
- **`FactoryPlugin` base class** created for ingestion plugins (4 migrated)
- **Module path helpers** consolidated
- **Legacy plugins** migrated to metadata pattern
- **Error handling** improved with structured types
- **`get_source_root()`** consolidated from 4 graph builders (~80 lines)
- **`is_test_path()`** consolidated from 4 files (~40 lines)
- **`filter_paths()`** consolidated from 4 files (~50 lines)

**Total lines removed in Phase 1+2:** ~870 lines

### Remaining (Phase 3 - Graph Builder Refinement)

| Item | Estimated Savings | Effort |
|------|-------------------|--------|
| `filter_mapping()` helper | ~40 lines | 30 min |
| `persist_rows()` helper | ~160 lines | 1 hour |
| `GraphBuilderPlugin` base class | ~80 lines | 2 hours |
| Common table loaders | ~40 lines | 1 hour |
| Step execution helper | ~30 lines | 30 min |
| **Total** | **~350 lines** | **~5 hours** |

### Remaining (Phase 4 - Infrastructure)

| Item | Effort |
|------|--------|
| Context hierarchy simplification | 2 hours |
| Export mixin + artifact support | 2 hours |
| Import organization | 2 hours |

---

## Appendix: Files with Remaining Consolidation Opportunities

### Files with `_persist_*` Functions

| File | Function Count | Can Use Generic |
|------|----------------|-----------------|
| `callgraph.py` | 2 | 1 (nodes only, edges have JSON) |
| `import_graph.py` | 2 | 2 |
| `cfg_dfg.py` | 3 | 3 |
| `goid.py` | 2 | 2 |
| `symbol_uses.py` | 1 | 1 |
| **Total** | **10** | **9** |

### Files with Dict Filtering

| File | Functions |
|------|-----------|
| `import_graph.py` | `_filter_paths_by_scope` (mapping) |
| `symbol_uses.py` | `_matches_scope`, `_filter_module_map`, `_filter_path_to_goid_map`, `_filter_occurrences` |

---

## Next Steps

1. **Immediate (Part 15):** Add `filter_mapping()` helper
2. **Short-term (Part 16):** Add `persist_rows()` helper
3. **Medium-term (Part 17):** Design and implement `GraphBuilderPlugin`
4. **Long-term:** Context and infrastructure improvements

The highest-impact remaining items are Parts 15-17, which would eliminate ~280 lines of duplication and standardize the graph builder implementation pattern.
