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

---

## Remaining Work & New Opportunities

### Part 12: Source Root Helper Consolidation (NEW - HIGH PRIORITY)

#### Problem

The `_get_source_root` function is **duplicated verbatim in 4 graph builder plugins**:

```python
# Identical in: callgraph.py, import_graph.py, cfg_dfg.py, goid.py
def _get_source_root(gateway: StorageGateway, repo: str, commit: str) -> Path | None:
    """Retrieve source root from core.snapshots."""
    try:
        snapshots = gateway.ibis.table("core.snapshots")
        expr = (
            snapshots.filter(
                cast("Any", snapshots.repo == repo) & cast("Any", snapshots.commit == commit)
            )
            .select(snapshots.source_root)
            .limit(1)
        )
        df = expr.execute()
        if not getattr(df, "empty", True):
            value = df.iloc[0][0]
            if value:
                return Path(str(value))
    except DuckDBError as exc:
        log.debug("...: Could not get source root: %s", exc)
    return None
```

**Files with duplication:**
- `graphs/builders/callgraph.py`
- `graphs/builders/import_graph.py`
- `graphs/builders/cfg_dfg.py`
- `graphs/builders/goid.py`

#### Solution

Move to `build/plugins/_helpers.py`:

```python
def get_source_root(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    *,
    fallback: Path | None = None,
) -> Path:
    """Retrieve source root from core.snapshots.

    Parameters
    ----------
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit SHA.
    fallback
        Fallback path if not found (defaults to Path.cwd()).

    Returns
    -------
    Path
        Source root path.
    """
    try:
        snapshots = gateway.ibis.table("core.snapshots")
        expr = (
            snapshots.filter(
                cast("Any", snapshots.repo == repo) & cast("Any", snapshots.commit == commit)
            )
            .select(snapshots.source_root)
            .limit(1)
        )
        df = expr.execute()
        if not getattr(df, "empty", True):
            value = df.iloc[0][0]
            if value:
                return Path(str(value))
    except DuckDBError:
        pass
    return fallback or Path.cwd()
```

**Estimated savings:** ~80 lines

---

### Part 13: Test Path Detection Consolidation (NEW - HIGH PRIORITY)

#### Problem

The `_is_test_path` function is **duplicated in 4 files** with identical logic:

```python
# Identical in: symbol_uses.py, cfg_dfg.py, goid.py, ingestion/helpers.py
def _is_test_path(path: str) -> bool:
    """Return True when the path looks like a test file."""
    lowered = path.lower()
    return (
        "tests/" in lowered
        or lowered.endswith("_test.py")
        or "/test_" in lowered
        or lowered.startswith("test_")
    )
```

**Files with duplication:**
- `graphs/builders/symbol_uses.py`
- `graphs/builders/cfg_dfg.py`
- `graphs/builders/goid.py`
- `ingestion/helpers.py`

#### Solution

Move to `build/plugins/_helpers.py` as a public utility:

```python
def is_test_path(path: str) -> bool:
    """Check whether a path appears to be a test file.

    Uses common Python test file naming conventions:
    - Files in a `tests/` directory
    - Files ending in `_test.py`
    - Files containing `/test_` in the path
    - Files starting with `test_`

    Parameters
    ----------
    path
        Relative file path to check.

    Returns
    -------
    bool
        True if the path matches test file patterns.
    """
    lowered = path.lower()
    return (
        "tests/" in lowered
        or lowered.endswith("_test.py")
        or "/test_" in lowered
        or lowered.startswith("test_")
    )
```

**Estimated savings:** ~40 lines

---

### Part 14: Path Filtering Consolidation (NEW - HIGH PRIORITY)

#### Problem

Multiple plugins have nearly identical path filtering functions with minor variations:

```python
# In callgraph.py:
def _filter_paths_by_scope(paths: list[str], scope_paths: list[str] | None) -> list[str]:
    if not scope_paths:
        return paths
    prefixes = tuple(scope_paths)
    return [path for path in paths if path.startswith(prefixes)]

# In cfg_dfg.py (adds test filtering):
def _filter_paths(paths: list[str], options: CfgDfgOptions) -> list[str]:
    filtered = list(paths)
    if options.scope_paths:
        prefixes = tuple(options.scope_paths)
        filtered = [path for path in filtered if path.startswith(prefixes)]
    if not options.include_test_files:
        filtered = [path for path in filtered if not _is_test_path(path)]
    return filtered

# In scip_plugin.py:
def _filter_paths(paths: list[str], scope_paths: list[str] | None) -> list[str]:
    if not scope_paths:
        return paths
    prefixes = tuple(scope_paths)
    return [path for path in paths if path.startswith(prefixes)]
```

**Files with variations:**
- `graphs/builders/callgraph.py`
- `graphs/builders/import_graph.py`
- `graphs/builders/cfg_dfg.py`
- `graphs/builders/goid.py`
- `ingestion/scip_plugin.py`

#### Solution

Create unified filter function in `build/plugins/_helpers.py`:

```python
def filter_paths(
    paths: Iterable[str],
    *,
    scope_paths: list[str] | None = None,
    include_tests: bool = True,
) -> list[str]:
    """Filter paths by scope and test inclusion.

    Parameters
    ----------
    paths
        Paths to filter.
    scope_paths
        Optional list of path prefixes to include. If None, all paths are included.
    include_tests
        Whether to include test files. Uses `is_test_path()` for detection.

    Returns
    -------
    list[str]
        Filtered list of paths.
    """
    result = list(paths)
    
    if scope_paths:
        prefixes = tuple(scope_paths)
        result = [path for path in result if path.startswith(prefixes)]
    
    if not include_tests:
        result = [path for path in result if not is_test_path(path)]
    
    return result
```

**Estimated savings:** ~50 lines

---

### Part 15: Persistence Pattern Consolidation (NEW - HIGH PRIORITY)

#### Problem

All graph builder plugins follow an identical persistence pattern:

```python
# Repeated in 6+ files with minor variations:
def _persist_X(
    gateway: StorageGateway,
    rows: list[XRow],
    repo: str,
    commit: str,
) -> int:
    if not rows:
        return 0
    gateway.policy.ensure_table("table.key")
    gateway.policy.delete_for_snapshot("table.key", repo=repo, commit=commit)
    gateway.policy.bulk_insert("table.key", [row.to_tuple() for row in rows])
    return len(rows)
```

**Files with this pattern:**
- `graphs/builders/callgraph.py` (2 functions: nodes, edges)
- `graphs/builders/import_graph.py` (2 functions: modules, edges)
- `graphs/builders/cfg_dfg.py` (3 functions: blocks, cfg_edges, dfg_edges)
- `graphs/builders/goid.py` (2 functions: goids, crosswalk)
- `graphs/builders/symbol_uses.py` (1 function: edges)

#### Solution

Create generic `persist_rows` helper in `build/plugins/_helpers.py`:

```python
from typing import Protocol, TypeVar

class RowWithTuple(Protocol):
    """Protocol for rows that can be converted to tuples."""
    def to_tuple(self) -> tuple[Any, ...]: ...

TRow = TypeVar("TRow", bound=RowWithTuple)

def persist_rows(
    gateway: StorageGateway,
    table_key: str,
    rows: Sequence[TRow],
    *,
    repo: str,
    commit: str,
) -> int:
    """Persist rows to a table with snapshot cleanup.

    Handles the common pattern of:
    1. Early return on empty input
    2. Ensure table exists
    3. Delete existing snapshot data
    4. Bulk insert new rows

    Parameters
    ----------
    gateway
        Storage gateway.
    table_key
        Fully-qualified table name (e.g., "graph.call_graph_edges").
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

Alternative: Support both `to_tuple()` and pre-converted tuples:

```python
def persist_rows(
    gateway: StorageGateway,
    table_key: str,
    rows: Sequence[TRow] | Sequence[tuple[Any, ...]],
    *,
    repo: str,
    commit: str,
    convert: Callable[[TRow], tuple[Any, ...]] | None = None,
) -> int:
    """Persist rows with optional conversion."""
    if not rows:
        return 0
    
    gateway.policy.ensure_table(table_key)
    gateway.policy.delete_for_snapshot(table_key, repo=repo, commit=commit)
    
    if convert:
        tuples = [convert(row) for row in rows]
    elif hasattr(rows[0], "to_tuple"):
        tuples = [row.to_tuple() for row in rows]  # type: ignore
    else:
        tuples = list(rows)  # Already tuples
    
    gateway.policy.bulk_insert(table_key, tuples)
    return len(tuples)
```

**Estimated savings:** ~100 lines

---

### Part 16: GraphBuilderPlugin Base Class (NEW - MEDIUM PRIORITY)

#### Problem

All 5 graph builder plugins share a common structure:

1. Load function/module index from tables
2. Filter paths by scope/tests
3. Get source root
4. Process files and collect results
5. Persist results
6. Return row counts

```python
# Common pattern in all graph builders:
async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
    opts = self.resolve_options(XOptions)
    snapshot = ctx.snapshot
    gateway, repo, commit = ctx.gateway, snapshot.repo, snapshot.commit

    try:
        function_index = load_function_index(gateway, repo=repo, commit=commit)
        paths = _filter_paths(function_index.paths(), opts)

        if not paths:
            log.info("...: No functions found, skipping")
            return TargetResult.succeeded(row_counts={"table": 0, ...})

        source_root = (
            snapshot.repo_root or _get_source_root(gateway, repo, commit) or Path.cwd()
        )
        
        # ... process and collect ...
        # ... persist ...
        
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
    - Loading function/module indices
    - Path filtering
    - Source root resolution
    - Error handling
    """

    @property
    def empty_row_counts(self) -> dict[str, int]:
        """Return row counts when no data to process."""
        return {table: 0 for table in self._core_metadata.produces_tables}

    def get_source_root(self, ctx: TargetExecutionContext) -> Path:
        """Get source root with fallbacks."""
        if ctx.snapshot.repo_root:
            return ctx.snapshot.repo_root
        return get_source_root(ctx.gateway, ctx.repo, ctx.commit, fallback=Path.cwd())

    def filter_paths(
        self,
        paths: list[str],
        opts: TOptions,
    ) -> list[str]:
        """Filter paths using options. Override for custom filtering."""
        scope_paths = getattr(opts, "scope_paths", None)
        include_tests = getattr(opts, "include_tests", True)
        include_test_files = getattr(opts, "include_test_files", True)
        return filter_paths(
            paths,
            scope_paths=scope_paths,
            include_tests=include_tests and include_test_files,
        )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute with standard error handling."""
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

**Estimated savings:** ~150 lines + standardized error handling

---

### Part 17: Snapshot Destructuring Pattern (NEW - LOW PRIORITY)

#### Problem

Nearly every plugin starts with the same destructuring pattern:

```python
# Repeated in ~20+ plugin execute() methods:
gateway, repo, commit = ctx.gateway, ctx.snapshot.repo, ctx.snapshot.commit
# or
gateway = ctx.gateway
repo = snapshot.repo
commit = snapshot.commit
```

#### Solution

Add convenience property to `TargetExecutionContext`:

```python
@dataclass(frozen=True)
class TargetExecutionContext:
    # ... existing fields ...

    @property
    def snapshot_key(self) -> tuple[str, str]:
        """Return (repo, commit) tuple."""
        return (self.snapshot.repo, self.snapshot.commit)
    
    @property
    def repo(self) -> str:
        """Shorthand for snapshot.repo."""
        return self.snapshot.repo
    
    @property
    def commit(self) -> str:
        """Shorthand for snapshot.commit."""
        return self.snapshot.commit
```

This is already partially done (`ctx.repo`, `ctx.commit` exist). Verify all plugins use them consistently.

---

### Part 7: Context Hierarchy Simplification

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

### Part 8: Native Target Pattern Improvements

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

### Part 11: Import Organization

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
| 1 | Part 2: Row Count Helper | Low | High | ✅ COMPLETE |
| 2 | Part 1: Plugin Migration | Medium | High | ✅ COMPLETE (28/28) |
| 3 | Part 3: Options Resolver | Low | Medium | ✅ COMPLETE |
| 4 | Part 4: Factory Plugin Base | Medium | Medium | ✅ COMPLETE |
| 5 | Part 5: Module Path Helpers | Low | Medium | ✅ COMPLETE |
| 6 | Part 6: Legacy Plugin Metadata | Low | Medium | ✅ COMPLETE |
| 7 | Part 10: Error Handling | Low | Medium | ✅ COMPLETE (key locations) |
| **8** | **Part 12: Source Root Helper** | **Low** | **High** | **Pending** |
| **9** | **Part 13: Test Path Detection** | **Low** | **Medium** | **Pending** |
| **10** | **Part 14: Path Filtering** | **Low** | **Medium** | **Pending** |
| **11** | **Part 15: Persistence Pattern** | **Medium** | **High** | **Pending** |
| **12** | **Part 16: GraphBuilderPlugin** | **Medium** | **High** | **Pending** |
| 13 | Part 17: Snapshot Destructuring | Low | Low | Pending |
| 14 | Part 7: Context Simplification | Medium | Medium | Pending |
| 15 | Part 8.1: Executor Artifacts | Low | Medium | Pending |
| 16 | Part 8.2: Export Mixin | Low | Low | Pending |
| 17 | Part 11: Import Organization | Medium | Low | Pending |

---

## Validation Checklist

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

### Completed (Phase 1)

- **28 plugins** migrated to `MetadataPlugin` (~700 lines removed)
- **Row count helper** created and used in 3+ plugins
- **Generic `resolve_options()`** added to `MetadataPlugin`
- **`FactoryPlugin` base class** created for ingestion plugins (4 migrated)
- **Module path helpers** consolidated
- **Legacy plugins** migrated to metadata pattern
- **Error handling** improved with structured types

### Remaining (Phase 2 - Graph Builder Consolidation)

| Item | Estimated Savings | Effort |
|------|-------------------|--------|
| `get_source_root` helper | ~80 lines | 30 min |
| `is_test_path` helper | ~40 lines | 15 min |
| `filter_paths` helper | ~50 lines | 20 min |
| `persist_rows` helper | ~100 lines | 30 min |
| `GraphBuilderPlugin` base class | ~150 lines | 2 hours |
| **Total** | **~420 lines** | **~3.5 hours** |

### Remaining (Phase 3 - Infrastructure)

| Item | Effort |
|------|--------|
| Context hierarchy simplification | 2 hours |
| Export mixin + artifact support | 2 hours |
| Import organization | 2 hours |

---

## Appendix: Duplication Analysis

### Files with `_get_source_root`

| File | Lines |
|------|-------|
| `graphs/builders/callgraph.py` | 235-266 |
| `graphs/builders/import_graph.py` | 60-93 |
| `graphs/builders/cfg_dfg.py` | 99-132 |
| `graphs/builders/goid.py` | 110-143 |

### Files with `_is_test_path`

| File | Lines |
|------|-------|
| `graphs/builders/symbol_uses.py` | 53-67 |
| `graphs/builders/cfg_dfg.py` | 62-76 |
| `graphs/builders/goid.py` | 70-84 |
| `ingestion/helpers.py` | 28-42 |

### Files with `_persist_*` Functions

| File | Function Count | Total Lines |
|------|----------------|-------------|
| `graphs/builders/callgraph.py` | 2 | ~60 |
| `graphs/builders/import_graph.py` | 2 | ~50 |
| `graphs/builders/cfg_dfg.py` | 3 | ~70 |
| `graphs/builders/goid.py` | 2 | ~50 |
| `graphs/builders/symbol_uses.py` | 1 | ~25 |
| **Total** | **10** | **~255** |

---

## Next Steps

1. **Immediate (Parts 12-15):** Create shared helpers in `_helpers.py`
2. **Short-term (Part 16):** Design and implement `GraphBuilderPlugin`
3. **Medium-term (Parts 7-8):** Context and executor improvements
4. **Long-term (Part 11):** Import organization and facade creation

The highest-impact items are Parts 12-16, which would eliminate ~420 lines of duplication from the graph builders and standardize their implementation pattern.
