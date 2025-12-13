# Ingestion Package Cleanup Plan

> **Generated:** 2025-12-13  
> **Package:** `codeintel.ingestion`  
> **Status:** Assessment Complete - Pending Implementation

## Executive Summary

The `codeintel.ingestion` package contains **50+ Python files** across 6 subpackages following a clean hexagonal architecture (ports/adapters/compute). The package is generally well-organized, but analysis identified several cleanup opportunities:

**Identified Issues:**
- 1 empty test directory (`tests/ingestion/engine/`)
- 1 potentially orphaned validation package (no production consumers)
- 3+ unused protocols/functions in tracker module
- 15+ potentially unused methods (vulture analysis)
- Worker infrastructure with unused exports

**Estimated Impact:**
- ~5 files potentially removable (validation package)
- ~1000 lines of dead code
- Cleaner public API surface

---

## Table of Contents

1. [Empty Test Directory](#1-empty-test-directory)
2. [Validation Package Assessment](#2-validation-package-assessment)
3. [Tracker Module Assessment](#3-tracker-module-assessment)
4. [Unused Methods (Vulture Analysis)](#4-unused-methods-vulture-analysis)
5. [Worker Infrastructure](#5-worker-infrastructure)
6. [Active Modules (Do Not Touch)](#6-active-modules-do-not-touch)
7. [Implementation Checklist](#7-implementation-checklist)
8. [Comparison with Other Packages](#8-comparison-with-other-packages)

---

## 1. Empty Test Directory

### Status: 🔴 Immediate Cleanup

**Location:** `tests/ingestion/engine/`

The directory contains only an empty `__init__.py`:

```python
"""Tests for ingestion engine components."""
```

**No actual tests exist** for the engine subpackage (ToolService, ToolPluginRegistry, tool plugins, etc.).

### Evidence

```bash
$ ls tests/ingestion/engine/
__init__.py
__pycache__/
```

### Recommendation

**Option A (Preferred):** Delete the empty directory if engine tests aren't planned.

**Option B:** Add meaningful tests for:
- `ToolService` orchestration
- `ToolPluginRegistry` registration
- Individual tool plugins (pyright, ruff, coverage, scip, pytest, pyrefly)

---

## 2. Validation Package Assessment

### Status: 🟡 Medium Priority - Potentially Orphaned

**Location:** `src/codeintel/ingestion/validation/`

The validation package exports a rich contract validation system but appears to have **zero production consumers** in `build/` or `cli/`.

### Package Contents

```
validation/
├── __init__.py          # Re-exports
├── findings.py          # Contract types (IngestContractSpec, etc.)
├── runner.py            # IngestContractValidator, run_ingest_validations
└── checks/
    ├── __init__.py      # Check re-exports
    ├── constraints.py   # Constraint checker functions
    └── database.py      # Database check functions
```

### Usage Analysis

| Component | Production Usage | Test Usage |
|-----------|------------------|------------|
| `run_ingest_validations()` | ❌ Not used in `build/` or `cli/` | ❌ None |
| `IngestContractValidator` | ❌ Only internal usage | ❌ None |
| `IngestContractSpec` | ❌ Only exports | ❌ None |
| `IngestValidationOptions` | ❌ Only exports | ❌ None |
| `check_*` functions | ❌ Only within validation module | ❌ None |
| `db_queries.py` helpers | ❌ Only used by validation checks | ❌ None |

### Evidence

```bash
# No production usage found
$ grep -r "run_ingest_validations\|IngestContractValidator" src/codeintel/build/ src/codeintel/cli/
# (no output)

# No test imports found
$ grep -r "from codeintel.ingestion.validation" tests/
# (no output)
```

### Dependencies

The validation package uses `db_queries.py` helpers which are **only** consumed by validation:

| Helper Function | External Usage |
|-----------------|----------------|
| `safe_count_with_scope` | ❌ Only validation |
| `safe_min_value` | ❌ Only validation |
| `safe_max_value` | ❌ Only validation |
| `safe_not_null_fraction` | ❌ Only validation |
| `safe_count_orphan_refs` | ❌ Only validation |

### Recommendation

1. **Confirm status:** Determine if validation was intended for future use or is legacy code
2. **If unused:** Remove entire `validation/` package (~5 files, ~1000 lines)
3. **If planned:** Document intended usage and add tests

---

## 3. Tracker Module Assessment

### Status: 🟡 Medium Priority - Partially Orphaned

**Location:** `src/codeintel/ingestion/tracker.py`

The tracker module contains both actively-used and orphaned components.

### Component Analysis

| Component | Status | Evidence |
|-----------|--------|----------|
| `ChangeTracker` | ✅ **ACTIVELY USED** | Used in `repo_scan.py`, `context.py` |
| `ChangeTrackerDatasetView` | ✅ Used | Via ChangeTracker |
| `run_incremental_ingest()` | ❌ **NOT USED** | No usage in `build/` |
| `IncrementalIngestOps` | ❌ **NOT USED** | Protocol without implementations |
| `SupportsFullRebuild` | ❌ **NOT USED** | Protocol without implementations |
| `IncrementalIngestPolicy` | ❌ Internal only | Only default values used |

### Evidence

```bash
# ChangeTracker IS used
$ grep -r "ChangeTracker" src/codeintel/build/
src/codeintel/build/plugins/ingestion/repo_scan.py:from codeintel.ingestion.tracker import ChangeTracker
src/codeintel/build/plugins/ingestion/repo_scan.py:        tracker = ChangeTracker.create(
src/codeintel/build/context.py:    from codeintel.ingestion.tracker import ChangeTracker

# But run_incremental_ingest and protocols are NOT used
$ grep -r "run_incremental_ingest\|IncrementalIngestOps\|SupportsFullRebuild" src/codeintel/build/
# (no output)
```

### Test Coverage

```bash
$ grep -r "run_incremental_ingest\|IncrementalIngestOps\|SupportsFullRebuild\|IncrementalIngestPolicy" tests/
tests/ingestion/test_change_tracker.py:6  # Only file with usage
```

The test file exists but only tests the tracker components that are actually used.

### Recommendation

1. **Keep:** `ChangeTracker`, `ChangeTrackerDatasetView` (actively used)
2. **Review:** `run_incremental_ingest` framework
   - If not planned for use, remove from exports
   - If planned, document the intended pattern
3. **Consider removal:** `IncrementalIngestOps`, `SupportsFullRebuild` protocols

---

## 4. Unused Methods (Vulture Analysis)

### Status: 🟡 Review Required

Running vulture at 60% confidence identified these potentially unused elements:

### Adapters

| File | Element | Confidence |
|------|---------|------------|
| `adapters/build_tool_adapter.py` | `run_pytest()` | 60% |
| `adapters/duckdb_storage.py` | `fetch_dataframe()` | 60% |
| `adapters/filesystem_discovery.py` | `_repo_root` attribute | 60% |
| `adapters/filesystem_discovery.py` | `iter_modules()` | 60% |
| `adapters/filesystem_discovery.py` | `file_exists()` | 60% |
| `adapters/tool_runner.py` | `from_runner()` | 60% |
| `adapters/tool_runner.py` | `run_pytest()` | 60% |

### Engine

| File | Element | Confidence |
|------|---------|------------|
| `engine/_scip_resolver.py` | `from_strings()` | 60% |
| `engine/_scip_resolver.py` | `build()` | 60% |
| `engine/infrastructure/runner.py` | `GIT` constant | 60% |
| `engine/infrastructure/runner.py` | `load_json()` | 60% |
| `engine/plugins.py` | `TIMEOUT` constant | 60% |
| `engine/pyright.py` | `parse_diagnostics()` | 60% |
| `engine/results.py` | `by_path()` | 60% |
| `engine/results.py` | `definitions_by_location()` | 60% |
| `engine/service.py` | `get_test_report()` | 60% |

### Infrastructure

| File | Element | Confidence |
|------|---------|------------|
| `infrastructure/ast_utils.py` | `from_tree()` | 60% |
| `infrastructure/ast_utils.py` | `lookup()` | 60% |
| `infrastructure/cst_utils.py` | `METADATA_DEPENDENCIES` | 60% |
| `infrastructure/cst_utils.py` | `on_visit()` | 60% |
| `infrastructure/cst_utils.py` | `on_leave()` | 60% |
| `infrastructure/workers.py` | `T` TypeVar | 60% |

### Ports

| File | Element | Confidence |
|------|---------|------------|
| `ports/change_detection.py` | `has_changes` property | 60% |
| `ports/change_detection.py` | `total_changed` property | 60% |
| `ports/discovery.py` | `file_exists()` | 60% |
| `ports/storage.py` | `fetch_dataframe()` | 60% |

### Recommendation

**Before removal, manually verify each:**
- Some may be protocol requirements (e.g., `on_visit`, `on_leave` for CST visitors)
- Some may be used via dynamic dispatch
- Some may be public API for external consumers

---

## 5. Worker Infrastructure

### Status: 🟢 Low Priority

**Location:** `src/codeintel/ingestion/infrastructure/workers.py`

### Usage Analysis

| Export | Production Usage |
|--------|------------------|
| `AST_WORKER_CONFIG` | ❌ Defined & exported only |
| `CST_WORKER_CONFIG` | ❌ Defined & exported only |
| `executor_factory` | ❌ Only used in orphaned `run_incremental_ingest` |
| `worker_pool` | ❌ Only exports |
| `create_executor` | ❌ Only exports |
| `WorkerConfig` | ❌ Only for config definition |
| `resolve_worker_count` | ✅ Possibly used |

### Evidence

```bash
$ grep -r "AST_WORKER_CONFIG\|CST_WORKER_CONFIG" src/codeintel/
# Only definitions and re-exports, no actual consumption
```

### Recommendation

These appear to be infrastructure for a parallelization pattern that isn't currently active:

1. If parallel processing is planned, document the intended usage
2. If not needed, consider removing from public exports
3. Keep `resolve_worker_count` if actively used

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

### Phase 1: Safe Cleanup (No Risk) ⬜

- [ ] Delete empty `tests/ingestion/engine/` directory
- [ ] Verify and remove obviously unused methods from vulture analysis

### Phase 2: Tracker Review ⬜

- [ ] Confirm `run_incremental_ingest` pattern is not planned for future use
- [ ] If confirmed unused:
  - [ ] Remove from `__all__` exports in `tracker.py`
  - [ ] Remove from `__all__` exports in `__init__.py`
  - [ ] Mark internal functions as private (prefix with `_`)
- [ ] If planned for future, add documentation

### Phase 3: Validation Package Review ⬜

- [ ] Determine if validation package is intended infrastructure or dead code
- [ ] **If dead code:**
  - [ ] Delete `src/codeintel/ingestion/validation/` directory
  - [ ] Remove validation exports from `src/codeintel/ingestion/__init__.py`
  - [ ] Remove unused `db_queries.py` helpers only used by validation
- [ ] **If intended for future:**
  - [ ] Add documentation explaining planned usage
  - [ ] Add basic tests to prevent rot

### Phase 4: Worker Config Review ⬜

- [ ] Verify worker configs are not needed by any consumers
- [ ] If unused, remove from public exports

### Phase 5: Vulture Method Cleanup ⬜

- [ ] Review each flagged method manually
- [ ] Remove confirmed dead code
- [ ] Document any methods kept for protocol compliance

---

## 8. Comparison with Other Packages

| Issue Type | Analytics | Graphs | Ingestion |
|------------|-----------|--------|-----------|
| Empty test dirs | 1 (ports/) ✅ | 2 (core/, runtime/) ✅ | 1 (engine/) ⬜ |
| Deprecated stubs | 1 (adapters/) ✅ | 1 (adapters/) ✅ | 0 |
| Unused protocols | 1 (GraphRuntimePort) ✅ | 1 (ParsingPort) ✅ | 2+ (IncrementalIngestOps, etc.) ⬜ |
| Orphaned modules | 0 | 0 | 1 (validation/) ⬜ |
| Unused aliases | 3 ✅ | 9 ✅ | 0 |
| Re-export packages | 1 (ports/) ✅ | - | 0 (ports actively used) |

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

## Related Documents

- [ANALYTICS_CLEANUP_PLAN.md](./ANALYTICS_CLEANUP_PLAN.md) - Completed
- [GRAPHS_CLEANUP_PLAN.md](./GRAPHS_CLEANUP_PLAN.md) - Completed
- [BUILD_CLEANUP_PLAN.md](./BUILD_CLEANUP_PLAN.md)
- [BUILD_CONSOLIDATION_PLAN.md](./BUILD_CONSOLIDATION_PLAN.md)
- [BUILD_REFINEMENT_PLAN.md](./BUILD_REFINEMENT_PLAN.md)

