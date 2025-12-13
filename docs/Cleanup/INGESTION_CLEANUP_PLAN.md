# Ingestion Package Cleanup Plan

> **Generated:** 2025-12-13  
> **Updated:** 2025-12-13 (Phases 1-8 completed)  
> **Package:** `codeintel.ingestion`  
> **Status:** Phases 1-8 Complete

## Executive Summary

The `codeintel.ingestion` package contains **50+ Python files** across 6 subpackages following a clean hexagonal architecture (ports/adapters/compute). After comprehensive cleanup, the package is well-organized with reduced duplication.

**Completed Cleanup (Phases 1-8):**
- ~~1 empty test directory (`tests/ingestion/engine/`)~~ ✅ Deleted
- ~~1 orphaned validation package (6 files)~~ ✅ Deleted  
- ~~4 unused tracker exports~~ ✅ Removed from public API
- ~~11 unused validation exports~~ ✅ Removed from public API
- ~~Worker config exports unused~~ ✅ Removed `AST_WORKER_CONFIG`, `CST_WORKER_CONFIG` from public API
- ~~Dead port methods~~ ✅ Removed `file_exists()` from port/adapter, `get_test_report()` from service
- ~~`_safe_relpath()` duplication~~ ✅ Consolidated to `infrastructure/paths.safe_relpath()`
- ~~Compute step boilerplate~~ ✅ Created `BaseExtractStep` base class, refactored 3 steps
- ~~Tool plugin NOT_FOUND duplication~~ ✅ Created `DiagnosticToolPlugin` base class, refactored 3 plugins

**Impact Achieved:**
- 7 files removed (validation package, empty test dir)
- ~370 lines of code consolidated via base classes
- ~60 lines of dead code removed
- Cleaner public API surface (15 unused exports removed)
- Two new reusable base classes (`BaseExtractStep`, `DiagnosticToolPlugin`)

**Remaining Consolidation Opportunities (Phase 9+):**
- 🔴 **Dead Port Interface:** `run_pytest()` defined but never called (~50 lines across 3 files)
- 🟡 **Deprecated Method:** `parse_diagnostics()` in pyright.py marked deprecated (~45 lines)
- 🟡 **Result Type Bloat:** Unused attributes in TestReport, ScipIndexResult (~100 lines)
- 🟢 **Infrastructure Dead Code:** `AstNodeIndex` class, SCIP resolver methods (~80 lines)
- 🟢 **Tracker Internal Dead Code:** `run_incremental_ingest` and related protocols (~150 lines)

**Total Remaining Cleanup Potential:** ~400+ lines

---

## Table of Contents

1. [Empty Test Directory](#1-empty-test-directory) ✅
2. [Validation Package Assessment](#2-validation-package-assessment) ✅
3. [Tracker Module Assessment](#3-tracker-module-assessment) ✅
4. [Unused Methods (Vulture Analysis)](#4-unused-methods-vulture-analysis) ⚡ Partially Complete
5. [Worker Infrastructure](#5-worker-infrastructure) ✅
6. [Active Modules (Do Not Touch)](#6-active-modules-do-not-touch)
7. [Implementation Checklist](#7-implementation-checklist)
8. [Comparison with Other Packages](#8-comparison-with-other-packages)
9. [Code Consolidation Opportunities](#9-code-consolidation-opportunities) ✅ Completed
10. [Remaining Cleanup Opportunities](#10-remaining-cleanup-opportunities) ⬜ NEW

---

## 1. Empty Test Directory

### Status: ✅ Complete

**Location:** `tests/ingestion/engine/` (DELETED)

The empty test directory has been removed.

### Action Taken

- Deleted `tests/ingestion/engine/__init__.py`
- Removed `tests/ingestion/engine/` directory

**Note:** If engine tests are needed in the future, recreate the directory with meaningful tests for:
- `ToolService` orchestration
- `ToolPluginRegistry` registration
- Individual tool plugins (pyright, ruff, coverage, scip, pytest, pyrefly)

---

## 2. Validation Package Assessment

### Status: ✅ Complete

**Location:** `src/codeintel/ingestion/validation/` (DELETED)

The validation package was confirmed as dead code with zero production consumers.

### Action Taken

- Deleted 6 files from `validation/` package:
  - `__init__.py`
  - `findings.py`
  - `runner.py`
  - `checks/__init__.py`
  - `checks/constraints.py`
  - `checks/database.py`
- Removed 11 validation exports from `src/codeintel/ingestion/__init__.py`

### Retained Items

- `db_queries.py` in `infrastructure/` - Contains useful query utilities with comprehensive tests
- Can be used for future validation or analytics if needed

### Impact

- ~1000 lines of dead code removed
- Cleaner import surface for the package

---

## 3. Tracker Module Assessment

### Status: ✅ Complete

**Location:** `src/codeintel/ingestion/tracker.py`

The tracker module's public API has been cleaned up to export only actively-used components.

### Action Taken

| Component | Status | Action |
|-----------|--------|--------|
| `ChangeTracker` | ✅ USED | Kept in exports |
| `ChangeTrackerDatasetView` | ✅ USED | Kept in exports |
| `run_incremental_ingest()` | ❌ NOT USED | Removed from `__all__` |
| `IncrementalIngestOps` | ❌ NOT USED | Removed from `__all__` |
| `SupportsFullRebuild` | ❌ NOT USED | Removed from `__all__` |
| `IncrementalIngestPolicy` | ❌ NOT USED | Removed from `__all__` |

### Current Public API

```python
# tracker.py __all__
__all__ = [
    "ChangeTracker",
    "ChangeTrackerDatasetView",
]
```

### Retained Internal Code

The internal implementations (`run_incremental_ingest`, protocols, etc.) remain in the file for potential future reactivation. See Section 10.4 for full removal if confirmed unused.

---

## 4. Unused Methods (Vulture Analysis)

### Status: ⚡ Partially Complete

Running vulture at 60% confidence identified potentially unused elements. Items marked ✅ have been removed; remaining items are documented in Section 10.

### Removed Items ✅

| File | Element | Action |
|------|---------|--------|
| `ports/discovery.py` | `file_exists()` | ✅ Removed |
| `adapters/filesystem_discovery.py` | `file_exists()` | ✅ Removed |
| `engine/service.py` | `get_test_report()` | ✅ Removed |

### Remaining Items (See Section 10)

#### Adapters
| File | Element | Status |
|------|---------|--------|
| `adapters/build_tool_adapter.py` | `run_pytest()` | ⬜ Section 10.1 |
| `adapters/duckdb_storage.py` | `fetch_dataframe()` | ⬜ Keep (Ibis integration) |
| `adapters/filesystem_discovery.py` | `_repo_root` attribute | ⬜ Section 10.7 |
| `adapters/filesystem_discovery.py` | `iter_modules()` | ⬜ Section 10.7 |
| `adapters/tool_runner.py` | `from_runner()` | ⬜ Section 10.7 |
| `adapters/tool_runner.py` | `run_pytest()` | ⬜ Section 10.1 |

#### Engine
| File | Element | Status |
|------|---------|--------|
| `engine/_scip_resolver.py` | `from_strings()` | ⬜ Section 10.6 |
| `engine/_scip_resolver.py` | `build()` | ⬜ Section 10.6 |
| `engine/infrastructure/runner.py` | `GIT` constant | ⬜ Section 10.7 |
| `engine/infrastructure/runner.py` | `load_json()` | ⬜ Section 10.7 |
| `engine/pyright.py` | `parse_diagnostics()` | ⬜ Section 10.2 |
| `engine/results.py` | `by_path()` | ⬜ Section 10.3 |
| `engine/results.py` | `definitions_by_location()` | ⬜ Section 10.3 |

#### Infrastructure
| File | Element | Status |
|------|---------|--------|
| `infrastructure/ast_utils.py` | `from_tree()`, `lookup()` | ⬜ Section 10.5 |
| `infrastructure/cst_utils.py` | `METADATA_DEPENDENCIES` | ⬜ Section 10.5 |
| `infrastructure/cst_utils.py` | `on_visit()`, `on_leave()` | ✅ Keep (LibCST protocol) |
| `infrastructure/workers.py` | `T` TypeVar | ⬜ Section 10.5 |

#### Ports
| File | Element | Status |
|------|---------|--------|
| `ports/change_detection.py` | `has_changes`, `total_changed` | ⬜ Review needed |
| `ports/tools.py` | `run_pytest()` | ⬜ Section 10.1 |
| `ports/storage.py` | `fetch_dataframe()` | ⬜ Keep (Ibis integration) |

### Notes on Kept Items

1. **`on_visit()`, `on_leave()`** - Required by LibCST visitor protocol
2. **`fetch_dataframe()`** - Planned for future Ibis integration
3. **`TIMEOUT`** - Used as enum value, not dead code

---

## 5. Worker Infrastructure

### Status: ✅ Complete

**Location:** `src/codeintel/ingestion/infrastructure/workers.py`

### Action Taken

Removed unused worker config exports from public API:

| Export | Action |
|--------|--------|
| `AST_WORKER_CONFIG` | ✅ Removed from `__init__.py` and `infrastructure/__init__.py` |
| `CST_WORKER_CONFIG` | ✅ Removed from `__init__.py` and `infrastructure/__init__.py` |
| `executor_factory` | Kept (internal use) |
| `worker_pool` | Kept (may be used) |
| `create_executor` | Kept (may be used) |
| `WorkerConfig` | Kept (type definition) |
| `resolve_worker_count` | Kept (may be used) |

**Note:** The constants themselves remain in `workers.py` for potential future use; only the public exports were removed.

---

## 6. Active Modules (Do Not Touch)

The following modules are **heavily used** and should NOT be modified:

### Core Adapters

| Module | Import Count | Notes |
|--------|--------------|-------|
| `adapters.DuckDBStorageAdapter` | 12+ | Primary storage adapter |
| `adapters.FilesystemDiscoveryAdapter` | 8+ | Module discovery |
| `adapters.HashChangeDetectionAdapter` | 6+ | Change detection |
| `adapters.BuildToolAdapter` | 4+ | Build integration |
| `adapters.ToolRunnerAdapter` | 10+ | Tool execution |

### Compute Steps (All Actively Used)

| Module | Notes |
|--------|-------|
| `compute.AstExtractStep` | AST parsing |
| `compute.CstExtractStep` | CST parsing |
| `compute.DocstringsExtractStep` | Docstring extraction |
| `compute.TypingIngestStep` | Type annotation analysis |
| `compute.CoverageIngestStep` | Coverage data |
| `compute.TestsIngestStep` | Test results |
| `compute.ScipIngestStep` | SCIP indexing |
| `compute.ConfigIngestStep` | Config files |
| `compute.RepoScanStep` | Repository scanning |

### Ports (All Actively Used)

| Module | Import Count | Notes |
|--------|--------------|-------|
| `ports.IngestStoragePort` | 52+ | Core storage protocol |
| `ports.IngestToolPort` | 25+ | Tool execution protocol |
| `ports.ModuleDiscoveryPort` | 15+ | Discovery protocol |
| `ports.ChangeDetectionPort` | 12+ | Change detection protocol |

### Engine

| Module | Import Count | Notes |
|--------|--------------|-------|
| `engine.ToolService` | 16+ | Tool orchestration |
| `engine.ToolPluginRegistry` | 9+ | Plugin management |
| `engine.results.*` | 20+ | Result types |

### Infrastructure

| Module | Import Count | Notes |
|--------|--------------|-------|
| `infrastructure.paths` | 53+ | Path utilities |
| `infrastructure.scanning` | 20+ | Source scanning |
| `tracker.ChangeTracker` | 28+ | Change detection |

---

## 7. Implementation Checklist

### Phase 1: Safe Cleanup (No Risk) ✅

- [x] Delete empty `tests/ingestion/engine/` directory
- [x] Remove `file_exists()` from port and adapter (verified unused)
- [x] Remove `get_test_report()` from `engine/service.py`

### Phase 2: Tracker Review ✅

- [x] Confirm `run_incremental_ingest` pattern is not planned for future use
- [x] Remove from `__all__` exports in `tracker.py`
- [x] Remove from `__all__` exports in `__init__.py`
- Note: Internal code kept intact for potential future reactivation

### Phase 3: Validation Package Review ✅

- [x] Determined validation package is dead code (no production consumers)
- [x] Deleted `src/codeintel/ingestion/validation/` directory (6 files)
- [x] Removed validation exports from `src/codeintel/ingestion/__init__.py`
- Note: Kept `db_queries.py` - contains useful utilities with comprehensive tests

### Phase 4: Worker Config Review ✅

- [x] Verified `AST_WORKER_CONFIG`, `CST_WORKER_CONFIG` not used by any consumers
- [x] Removed from public exports in `__init__.py`
- [x] Removed from public exports in `infrastructure/__init__.py`
- Note: Constants kept internally in `workers.py` for potential future use

### Phase 5: Vulture Method Cleanup ✅ (Partial)

- [x] Removed `file_exists()` from `ports/discovery.py` and `adapters/filesystem_discovery.py`
- [x] Removed `get_test_report()` from `engine/service.py`
- [x] Removed corresponding tests from `tests/ingestion/test_tools.py`
- [ ] Remaining items deferred to Phase 9+ (see Section 10)

### Phase 6: DRY Violation Fix ✅

- [x] Added `safe_relpath()` to `infrastructure/paths.py`
- [x] Updated imports in `engine/pyright.py`
- [x] Updated imports in `engine/pyrefly.py`
- [x] Updated imports in `engine/ruff.py`
- [x] Updated imports in `engine/coverage.py`
- [x] Removed local `_safe_relpath()` definitions from all 4 files

### Phase 7: Compute Step Consolidation ✅

- [x] Created `BaseExtractStep` in `compute/base.py`
- [x] Added `_iter_python_sources()` helper method
- [x] Added `_write_and_count()` helper method
- [x] Refactored `AstExtractStep` to use base class
- [x] Refactored `CstExtractStep` to use base class
- [x] Refactored `DocstringsExtractStep` to use base class
- Note: `TypingIngestStep`, `CoverageIngestStep` have different signatures (require `IngestToolPort`)

### Phase 8: Tool Plugin Consolidation ✅

- [x] Created `DiagnosticToolPlugin` base class in `engine/plugins.py`
- [x] Added `tool_name` class variable and `_not_found_result()` method
- [x] Refactored `PyrightPlugin` to use base class
- [x] Refactored `PyreflyPlugin` to use base class
- [x] Refactored `RuffPlugin` to use base class
- Note: `CoveragePlugin`, `PytestPlugin`, `ScipPlugin` use different result types

---

## 8. Comparison with Other Packages

| Issue Type | Analytics | Graphs | Ingestion |
|------------|-----------|--------|-----------|
| Empty test dirs | 1 (ports/) ✅ | 2 (core/, runtime/) ✅ | 1 (engine/) ✅ |
| Deprecated stubs | 1 (adapters/) ✅ | 1 (adapters/) ✅ | 1 (parse_diagnostics) ⬜ |
| Unused protocols | 1 (GraphRuntimePort) ✅ | 1 (ParsingPort) ✅ | 4 (tracker protocols) ✅ |
| Orphaned modules | 0 | 0 | 1 (validation/) ✅ |
| Unused exports | 3 ✅ | 9 ✅ | 15 ✅ |
| Re-export packages | 1 (ports/) ✅ | - | 0 (ports actively used) |
| Base class consolidation | 0 | 0 | 2 (BaseExtractStep, DiagnosticToolPlugin) ✅ |
| DRY violations | 0 | 0 | 1 (safe_relpath) ✅ |

---

## Verification Commands

After implementing changes, run:

```bash
# Type checking
uv run pyright --warnings --pythonversion=3.13
uv run pyrefly check

# Linting
uv run ruff check --fix

# Tests
uv run pytest tests/ingestion/ -q

# Dead code check
uv run vulture src/codeintel/ingestion --min-confidence 80
```

---

## 9. Code Consolidation Opportunities

### Status: ✅ COMPLETED

The following consolidation opportunities were identified and implemented.

---

### 9.1 DRY Violation: `_safe_relpath` Function ✅ COMPLETED

**Impact:** ~60 lines consolidated

- Added `safe_relpath()` to `infrastructure/paths.py`
- Updated 4 engine plugins to use the shared function
- Removed local `_safe_relpath()` definitions from:
  - `engine/pyright.py`
  - `engine/pyrefly.py`
  - `engine/ruff.py`
  - `engine/coverage.py`

---

### 9.2 Compute Step Pattern Duplication ✅ COMPLETED

**Impact:** ~150 lines consolidated

Created `BaseExtractStep` base class in `compute/base.py`:

```python
class BaseExtractStep:
    """Base class for module extraction steps with port injection."""
    
    _storage: IngestStoragePort
    _discovery: ModuleDiscoveryPort
    
    def __init__(self, storage: IngestStoragePort, discovery: ModuleDiscoveryPort) -> None: ...
    def _iter_python_sources(self, modules: Sequence[ModuleRecord]) -> Iterator[tuple[ModuleRecord, str]]: ...
    def _write_and_count(self, table_key: str, rows: Sequence[Sequence[object]], *, repo: str, commit: str) -> dict[str, int]: ...
```

Refactored steps to inherit from `BaseExtractStep`:
- `AstExtractStep` ✅
- `CstExtractStep` ✅
- `DocstringsExtractStep` ✅

**Note:** `TypingIngestStep` and `CoverageIngestStep` have different signatures (require `IngestToolPort`) and were not refactored.

---

### 9.3 Tool Plugin Pattern Duplication ✅ COMPLETED

**Impact:** ~100 lines consolidated

Created `DiagnosticToolPlugin` base class in `engine/plugins.py`:

```python
@dataclass
class DiagnosticToolPlugin:
    """Base class for diagnostic tool plugins (pyright, pyrefly, ruff)."""
    
    tool_name: ClassVar[ToolName]
    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata
    
    def _not_found_result(self) -> ToolPluginResult: ...
```

Refactored plugins to inherit from `DiagnosticToolPlugin`:
- `PyrightPlugin` ✅
- `PyreflyPlugin` ✅
- `RuffPlugin` ✅

**Note:** `CoveragePlugin`, `PytestPlugin`, `ScipPlugin` use different result types and were not refactored.

---

### 9.4-9.7 Deferred Items

The following items from the original plan have been deferred to Phase 9+ (see Section 10):
- Unused port interface methods
- Result type consolidation
- AST/CST utility consolidation
- SCIP resolver dead code

---

## Consolidation Summary

| Item | Status | Lines Saved |
|------|--------|-------------|
| `_safe_relpath` consolidation | ✅ Complete | ~60 |
| Compute step base class | ✅ Complete | ~150 |
| Tool plugin base class | ✅ Complete | ~100 |
| Port methods cleanup | ⬜ Partial | ~20 (file_exists removed) |

**Total Lines Saved:** ~330

---

## 10. Remaining Cleanup Opportunities

### Status: ⬜ Identified - Pending Implementation

After completing Phases 1-8, the following opportunities remain for further cleanup. These are lower-priority items that emerged from post-implementation vulture analysis.

---

### 10.1 Dead Port Interface: `run_pytest()` (HIGH PRIORITY)

**Impact:** ~50 lines across 3 files

The `run_pytest()` method is defined in the port protocol and implemented in two adapters, but **never called anywhere**.

| File | Lines | Notes |
|------|-------|-------|
| `ports/tools.py` | 397-413 | Protocol definition |
| `adapters/tool_runner.py` | 388-430 | Implementation |
| `adapters/build_tool_adapter.py` | 253-280 | Implementation |

**Evidence:**
```bash
$ grep -r "\.run_pytest\(" src/codeintel/build/ src/codeintel/cli/
# (no output - never called)
```

**Recommendation:**
1. Confirm pytest integration uses `PytestPlugin` directly (it does)
2. Remove `run_pytest()` from port and adapters
3. Remove related test cases if any

---

### 10.2 Deprecated Method: `parse_diagnostics()` (MEDIUM PRIORITY)

**Impact:** ~45 lines

The `PyrightPlugin.parse_diagnostics()` static method is explicitly marked as deprecated:

```python
# engine/pyright.py:161-212
@staticmethod
def parse_diagnostics(result: ToolRunResult) -> dict[str, int]:
    """
    Parse pyright JSON from stdout into path -> error_count mapping.

    Deprecated: Use the parsed field on ToolPluginResult instead.
    ...
    """
```

**Recommendation:**
1. Verify no external consumers depend on this method
2. Remove the deprecated method
3. Update any documentation referencing it

---

### 10.3 Result Type Unused Attributes (MEDIUM PRIORITY)

**Impact:** ~100 lines of unused data structure fields

Vulture analysis reveals many dataclass fields are populated but never consumed:

| File | Type | Unused Fields |
|------|------|---------------|
| `engine/results.py:58` | `FileDiagnosticCount` | `warning_count` |
| `engine/results.py:193` | `CoverageReport` | `coverage_ratio` property |
| `engine/results.py:279` | `ScipIndexResult` | `by_path()` method |
| `engine/results.py:400-404` | `TestReport` | `passed_count`, `failed_count`, `skipped_count`, `total_duration_s` |
| `engine/results.py:628-629` | `ScipIndexResult` | `definition_count`, `reference_count` |
| `engine/results.py:703` | `ScipIndexResult` | `definitions_by_location()` method |

**Recommendation:**
1. Review if these attributes are intended for future analytics
2. If yes, document the planned usage
3. If no, remove unused fields to simplify the data model

---

### 10.4 Tracker Internal Dead Code (MEDIUM PRIORITY)

**Impact:** ~150 lines

The tracker module retains internal implementations that are no longer exported or used:

| Element | Lines | Notes |
|---------|-------|-------|
| `run_incremental_ingest()` | 417-490 | Function, never called |
| `IncrementalIngestOps` | 280-320 | Protocol, no implementations |
| `SupportsFullRebuild` | 320-340 | Protocol, no implementations |
| `IncrementalIngestPolicy` | 69-80 | Dataclass, only default used |
| `ChangeTracker.create()` | 126-150 | Factory method, never called |

**Recommendation:**
1. Document if incremental ingest pattern is planned for future
2. If not planned, delete the entire incremental framework
3. Keep only `ChangeTracker`, `ChangeTrackerDatasetView` (actively used)

---

### 10.5 Infrastructure Dead Code (LOW PRIORITY)

**Impact:** ~80 lines

| File | Element | Notes |
|------|---------|-------|
| `infrastructure/ast_utils.py:20-47` | `AstSpanIndex.from_tree()`, `lookup()` | Methods never called |
| `infrastructure/cst_utils.py:50` | `METADATA_DEPENDENCIES` | Constant never used |
| `infrastructure/workers.py:23` | `T` TypeVar | Never used in generic |
| `infrastructure/workers.py:48` | `executor_kind` | Field never accessed |

**Note:** `on_visit()` and `on_leave()` in `cst_utils.py` are LibCST protocol requirements and should NOT be removed.

**Recommendation:**
1. Remove unused `AstSpanIndex` methods if class isn't used externally
2. Remove unused constants and TypeVars
3. Keep CST visitor lifecycle methods

---

### 10.6 SCIP Resolver Dead Code (LOW PRIORITY)

**Impact:** ~40 lines

**Location:** `engine/_scip_resolver.py`

| Element | Lines | Notes |
|---------|-------|-------|
| `ScipPathConfig.from_strings()` | 55-75 | Factory method, never called |
| `ScipResolverInput.build()` | 134-160 | Builder method, never called |

**Recommendation:**
Review if SCIP path resolution uses these methods or alternative patterns; remove if unused.

---

### 10.7 Adapter Dead Code (LOW PRIORITY)

**Impact:** ~60 lines

| File | Element | Notes |
|------|---------|-------|
| `adapters/duckdb_storage.py:178` | `fetch_dataframe()` | Port method never called |
| `adapters/filesystem_discovery.py:51` | `_repo_root` attribute | Stored but never accessed |
| `adapters/filesystem_discovery.py:94` | `iter_modules()` | Method never called |
| `adapters/tool_runner.py:109` | `from_runner()` | Factory method never called |
| `engine/infrastructure/runner.py:31` | `GIT` constant | Never used |
| `engine/infrastructure/runner.py:257` | `load_json()` | Method never called |

**Recommendation:**
1. `fetch_dataframe()` - Keep if Ibis integration is planned
2. `_repo_root` - Remove attribute if unused
3. `iter_modules()` - Remove if `discover_modules()` is the preferred API
4. Others - Remove if confirmed unused

---

## Remaining Cleanup Priority Summary

| Priority | Issue | Est. Lines | Risk |
|----------|-------|------------|------|
| 🔴 HIGH | `run_pytest()` dead interface | ~50 | Low |
| 🟡 MEDIUM | `parse_diagnostics()` deprecated | ~45 | Low |
| 🟡 MEDIUM | Result type unused fields | ~100 | Low |
| 🟡 MEDIUM | Tracker internal dead code | ~150 | Medium |
| 🟢 LOW | Infrastructure dead code | ~80 | Low |
| 🟢 LOW | SCIP resolver dead code | ~40 | Low |
| 🟢 LOW | Adapter dead code | ~60 | Low |

**Total Remaining Cleanup Potential:** ~525 lines

---

## Phase 9+ Implementation Checklist

### Phase 9: Dead Port Interface ⬜

- [ ] Remove `run_pytest()` from `ports/tools.py`
- [ ] Remove `run_pytest()` from `adapters/tool_runner.py`
- [ ] Remove `run_pytest()` from `adapters/build_tool_adapter.py`
- [ ] Update any related test mocks

### Phase 10: Deprecated Method Removal ⬜

- [ ] Verify no external usage of `PyrightPlugin.parse_diagnostics()`
- [ ] Remove the method from `engine/pyright.py`

### Phase 11: Tracker Cleanup ⬜

- [ ] Confirm incremental ingest is not planned
- [ ] Remove `run_incremental_ingest()` function
- [ ] Remove `IncrementalIngestOps` protocol
- [ ] Remove `SupportsFullRebuild` protocol
- [ ] Remove `ChangeTracker.create()` if unused

### Phase 12: Infrastructure Cleanup ⬜

- [ ] Remove unused `AstSpanIndex` methods
- [ ] Remove `METADATA_DEPENDENCIES` constant
- [ ] Remove unused TypeVar and dataclass fields
- [ ] Remove SCIP resolver dead code
- [ ] Remove adapter dead code (except `fetch_dataframe`)

---

## Related Documents

- [ANALYTICS_CLEANUP_PLAN.md](./ANALYTICS_CLEANUP_PLAN.md) - Completed
- [GRAPHS_CLEANUP_PLAN.md](./GRAPHS_CLEANUP_PLAN.md) - Completed
- [BUILD_CLEANUP_PLAN.md](./BUILD_CLEANUP_PLAN.md)
- [BUILD_CONSOLIDATION_PLAN.md](./BUILD_CONSOLIDATION_PLAN.md)
- [BUILD_REFINEMENT_PLAN.md](./BUILD_REFINEMENT_PLAN.md)

