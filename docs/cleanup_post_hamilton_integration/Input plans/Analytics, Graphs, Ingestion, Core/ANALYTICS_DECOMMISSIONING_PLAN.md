# Analytics Package Post-Hamilton Decommissioning Plan

> **Generated:** 2025-12-16  
> **Package:** `codeintel.analytics`  
> **Status:** Planning Phase  
> **Priority:** High - Technical Debt Reduction

## Executive Summary

Following the Hamilton integration, the `analytics` package contains significant dead code, duplicate implementations, and legacy patterns that should be decommissioned. This document provides a comprehensive inventory of cleanup opportunities with specific file paths, code snippets, and migration strategies.

**Key Findings:**
- **1 empty directory** ready for immediate deletion
- **2 orphaned modules** with zero imports (safe to delete)
- **2 major code duplication cases** between orchestration and compute layers (~1,200 lines)
- **6+ constant/dataclass duplications** violating DRY
- **Multiple legacy persistence patterns** that should use Hamilton materializers exclusively

**Estimated Impact:**
- ~800-1,200 lines of dead/duplicate code can be removed
- Clearer separation between pure compute and Hamilton orchestration
- Reduced maintenance burden and confusion for contributors

---

## Table of Contents

1. [Immediate Deletions (Safe)](#1-immediate-deletions-safe)
2. [Orphaned Modules Analysis](#2-orphaned-modules-analysis)
3. [Duplicate Code Patterns](#3-duplicate-code-patterns)
4. [Legacy Persistence Patterns](#4-legacy-persistence-patterns)
5. [Constant and Dataclass Duplication](#5-constant-and-dataclass-duplication)
6. [Stale Documentation](#6-stale-documentation)
7. [Implementation Plan](#7-implementation-plan)
8. [Verification Checklist](#8-verification-checklist)

---

## 1. Immediate Deletions (Safe)

These items can be deleted immediately with no risk of breaking functionality.

### 1.1 Empty `runtime/` Directory

**Path:** `src/codeintel/analytics/runtime/`

**Current State:**
```
analytics/runtime/
└── __pycache__/    # Only contains bytecode cache
```

**Evidence of Non-Use:**
```bash
# No imports found anywhere in codebase
grep -r "analytics\.runtime\|from codeintel\.analytics import runtime" src/
# Result: No matches
```

**Action:** Delete entire directory

```bash
rm -rf src/codeintel/analytics/runtime/
```

**Risk Level:** ✅ None - directory is empty

---

## 2. Orphaned Modules Analysis

### 2.1 `graphs/plugin_catalog.py` (268 lines)

**Path:** `src/codeintel/analytics/graphs/plugin_catalog.py`

**Purpose:** Generates plugin documentation catalogs from the build registry.

**Evidence of Non-Use:**
```bash
# Search for any imports
grep -r "from codeintel\.analytics\.graphs\.plugin_catalog\|from codeintel\.analytics\.graphs import.*plugin_catalog" src/
# Result: No matches

grep -r "build_plugin_catalog\|render_plugin_catalog_markdown\|write_plugin_catalog" src/
# Result: Only found in the file itself
```

**Module Contents:**
```python
# Key exports that are never imported:
def build_plugin_catalog() -> dict[str, Any]: ...
def render_plugin_catalog_markdown(catalog: dict[str, Any] | None = None) -> str: ...
def write_plugin_catalog(path: Path) -> None: ...
def write_plugin_catalog_markdown(path: Path, catalog: dict[str, Any] | None = None) -> None: ...
def write_plugin_catalog_html(path: Path, catalog: dict[str, Any] | None = None) -> None: ...
```

**Analysis:**
- Uses `codeintel.build.unified_registry` directly
- Functionality likely superseded by build-layer catalog generation
- The `scripts/render_graph_plugin_catalog.py` exists but imports from different location

**Action:** Delete file after confirming no scripts use it

```bash
rm src/codeintel/analytics/graphs/plugin_catalog.py
```

**Risk Level:** ⚠️ Low - verify no CLI/scripts reference it

---

### 2.2 `graphs/contracts.py` (415 lines)

**Path:** `src/codeintel/analytics/graphs/contracts.py`

**Purpose:** Lightweight contract checking helpers for graph metric plugins.

**Evidence of Non-Use:**
```bash
# Search for any imports
grep -r "from codeintel\.analytics\.graphs\.contracts\|from codeintel\.analytics\.graphs import.*contracts" src/
# Result: No matches

grep -r "PluginContractResult\|run_contract_checkers\|assert_table_not_empty\|table_not_empty_checker" src/
# Result: Only found in the file itself
```

**Module Contents:**
```python
# Key exports that are never imported:
@dataclass(frozen=True)
class PluginContractResult:
    name: str
    status: Literal["passed", "failed", "soft_failed"]
    message: str | None = None

def run_contract_checkers(*, ctx: _ContractContext, checkers: tuple[ContractChecker, ...]) -> tuple[PluginContractResult, ...]: ...
def assert_table_not_empty(gateway, *, table, repo, commit, name=None) -> PluginContractResult: ...
def assert_table_exists(gateway, *, table, name=None) -> PluginContractResult: ...
def assert_columns_present(gateway, *, table, expected_columns, name=None) -> PluginContractResult: ...
def assert_not_null_fraction(gateway, *, snapshot, spec) -> PluginContractResult: ...
def table_not_empty_checker(table, *, name=None) -> ContractChecker: ...
def table_exists_checker(table, *, name=None) -> ContractChecker: ...
def columns_present_checker(table, *, expected_columns, name=None) -> ContractChecker: ...
def not_null_fraction_checker(table, *, column, min_fraction, name=None) -> ContractChecker: ...
```

**Analysis:**
- Contains contract checking infrastructure that was likely planned but never adopted
- Build layer has its own contract system (`build/hamilton/contracts/`)
- Hard-coded table schemas in `SAFE_TABLE_COLUMNS` dict suggests this was a prototype

**Action:** Delete file

```bash
rm src/codeintel/analytics/graphs/contracts.py
```

**Risk Level:** ✅ None - completely orphaned

---

## 3. Duplicate Code Patterns

### 3.1 Semantic Roles Duplication (~1,000 lines total)

**Critical Issue:** Nearly identical implementations exist in two locations.

#### Location A: `analytics/semantic_roles/core.py` (708 lines)

**Path:** `src/codeintel/analytics/semantic_roles/core.py`

**Contains:**
- I/O operations (database reads/writes)
- Pure classification logic (duplicated)
- Direct `backend.bulk_insert()` calls

#### Location B: `analytics/compute/semantic_roles/classification.py` (454 lines)

**Path:** `src/codeintel/analytics/compute/semantic_roles/classification.py`

**Contains:**
- Pure classification logic only
- No I/O operations
- Designed for Hamilton integration

#### Side-by-Side Comparison of Duplicated Elements:

| Element | `core.py` | `classification.py` | Notes |
|---------|-----------|---------------------|-------|
| `ROLE_THRESHOLD` | Line 30 | Line 24 | Identical: `0.35` |
| `SERVICE_FAN_IN_THRESHOLD` | Line 31 | Line 25 | Identical: `5` |
| `SERVICE_FAN_OUT_THRESHOLD` | Line 32 | Line 26 | Identical: `5` |
| `HELPER_LOC_THRESHOLD` | Line 33 | Line 27 | Identical: `20` |
| `FunctionContext` dataclass | Lines 36-99 | Lines 30-88 | Identical fields |
| `RoleAccumulator` dataclass | Lines 101-149 | Lines 91-137 | Identical implementation |
| `RoleArtifacts` dataclass | Lines 152-163 | Lines 140-150 | Identical fields |
| `ModuleRecord` dataclass | Lines 165-171 | Lines 153-158 | Identical fields |
| `classify_function_role()` | Lines 455-466 | Lines 161-187 | Same logic |
| `_score_tests()` | Lines 469-477 | Lines 291-299 | Identical |
| `_score_api_handlers()` | Lines 480-500 | Lines 302-322 | Identical |
| `_score_cli_commands()` | Lines 528-545 | Lines 350-367 | Identical |
| `_score_repositories()` | Lines 548-556 | Lines 370-378 | Identical |
| `_score_services()` | Lines 559-565 | Lines 381-387 | Identical |
| `_score_validators()` | Lines 568-578 | Lines 390-400 | Identical |
| `_score_config_loaders()` | Lines 581-585 | Lines 403-407 | Identical |
| `_score_helpers()` | Lines 588-594 | Lines 410-416 | Identical |
| `_score_module_tags()` | Lines 597-606 | Lines 419-428 | Identical |
| `_score_module_hints()` | Lines 609-612 | Lines 431-434 | Identical |

#### Current Usage:

```python
# Hamilton native module uses the I/O version:
# src/codeintel/build/hamilton/native/analytics/semantic_roles.py
from codeintel.analytics.semantic_roles import compute_semantic_roles
```

#### Recommended Action:

1. **Delete** `analytics/semantic_roles/core.py` entirely
2. **Update** Hamilton native module to use pure functions from `compute/semantic_roles/classification.py`
3. **Move** I/O operations into Hamilton materializer only

**Migration Path:**

```python
# BEFORE (in Hamilton native module):
from codeintel.analytics.semantic_roles import compute_semantic_roles
# ... calls compute_semantic_roles() which does DB writes internally

# AFTER:
from codeintel.analytics.compute.semantic_roles import (
    classify_function_role,
    classify_modules,
    FunctionContext,
    RoleArtifacts,
)
# ... Hamilton node does pure computation
# ... Materializer handles DB writes
```

---

### 3.2 Dependencies Core Duplication (~400 lines overlap)

**Locations:**
- `analytics/dependencies/core.py` (620+ lines) - Mixed I/O and logic
- `analytics/compute/dependencies/classification.py` (262 lines) - Pure logic
- `analytics/compute/dependencies/detection.py` (200+ lines) - Pure logic

#### Duplicated Elements:

| Element | `core.py` | `compute/` location | Lines Saved |
|---------|-----------|---------------------|-------------|
| `SEVERITY_SCORES` dict | Line 80-86 | `classification.py:18-24` | 7 |
| `CALLSITE_MEDIUM_THRESHOLD` | Line 79 | `classification.py:26` | 1 |
| `DependencyModePattern` dataclass | Lines 89-99 | `classification.py:29-57` | 11 |
| `LibraryPattern` dataclass | Lines 102-113 | `classification.py:85-113` | 12 |
| `DependencyCall` dataclass | Lines 116-129 | `detection.py` | 14 |

#### Recommended Action:

1. **Remove** duplicate constants from `core.py`
2. **Import** from `compute/dependencies/` instead
3. **Eventually** delete persistence logic from `core.py` when Hamilton migration complete

---

## 4. Legacy Persistence Patterns

### 4.1 Direct Database Writes (Anti-Pattern)

The following modules contain direct `backend.bulk_insert()` or `backend.delete_for_snapshot()` calls that should be migrated to Hamilton materializers:

#### `analytics/semantic_roles/core.py`

```python
# Lines 226-247 - Direct DB writes
backend.delete_for_snapshot(
    "analytics.semantic_roles_functions",
    repo=snapshot.repo,
    commit=snapshot.commit,
)
if fn_rows:
    backend.bulk_insert("analytics.semantic_roles_functions", fn_rows)

# ... similar pattern for semantic_roles_modules
```

**Migration:** Use `build/hamilton/native/analytics/semantic_roles.py` materializer exclusively.

---

#### `analytics/graphs/graph_metrics.py`

```python
# Lines 290-298 - Uses insert_analytics_rows helper
insert_analytics_rows(
    gateway,
    contract,
    validated_rows,
    delete_scope=DeleteScope(repo=snapshot.repo, commit=snapshot.commit),
    scope=f"{snapshot.repo}@{snapshot.commit}",
)
```

**Note:** This uses a helper function which is acceptable, but should eventually be Hamilton-only.

---

#### `analytics/functions/metrics.py`

```python
# Lines 626-650 - Direct persistence in persist_function_analytics()
backend.ensure_table(metrics_contract.table_key)
backend.delete_for_snapshot(...)
backend.bulk_insert_mappings(metrics_contract.table_key, validated_metrics)
# ... similar for types and validation tables
```

**Migration:** The Hamilton native module `build/hamilton/native/analytics/function_metrics.py` should be the only write path.

---

### 4.2 Deprecated Method References

#### `analytics/parsing/compute.py` (Lines 6-7, 89, 118)

```python
"""
Use these functions instead of the deprecated `flush()` methods on the reporter
classes.
"""

# Line 89:
"""Use this function instead of `FunctionValidationReporter.flush()` to..."""

# Line 118:
"""Use this function instead of `GraphValidationReporter.flush()` to..."""
```

**Issue:** The `flush()` methods no longer exist on these reporters. They now have `to_rows()` methods.

**Action:** Update docstrings to remove references to non-existent `flush()` methods.

---

## 5. Constant and Dataclass Duplication

### 5.1 Complete Duplication Inventory

| Constant/Class | Location 1 | Location 2 | Action |
|----------------|------------|------------|--------|
| `ROLE_THRESHOLD = 0.35` | `semantic_roles/core.py:30` | `compute/semantic_roles/classification.py:24` | Delete from `core.py` |
| `SERVICE_FAN_IN_THRESHOLD = 5` | `semantic_roles/core.py:31` | `compute/semantic_roles/classification.py:25` | Delete from `core.py` |
| `SERVICE_FAN_OUT_THRESHOLD = 5` | `semantic_roles/core.py:32` | `compute/semantic_roles/classification.py:26` | Delete from `core.py` |
| `HELPER_LOC_THRESHOLD = 20` | `semantic_roles/core.py:33` | `compute/semantic_roles/classification.py:27` | Delete from `core.py` |
| `SEVERITY_SCORES` dict | `dependencies/core.py:80-86` | `compute/dependencies/classification.py:18-24` | Delete from `core.py` |
| `CALLSITE_MEDIUM_THRESHOLD = 10` | `dependencies/core.py:79` | `compute/dependencies/classification.py:26` | Delete from `core.py` |
| `FunctionContext` dataclass | `semantic_roles/core.py:36-99` | `compute/semantic_roles/classification.py:30-88` | Delete from `core.py` |
| `RoleAccumulator` dataclass | `semantic_roles/core.py:101-149` | `compute/semantic_roles/classification.py:91-137` | Delete from `core.py` |
| `RoleArtifacts` dataclass | `semantic_roles/core.py:152-163` | `compute/semantic_roles/classification.py:140-150` | Delete from `core.py` |
| `ModuleRecord` dataclass | `semantic_roles/core.py:165-171` | `compute/semantic_roles/classification.py:153-158` | Delete from `core.py` |
| `DependencyModePattern` dataclass | `dependencies/core.py:89-99` | `compute/dependencies/classification.py:29-57` | Delete from `core.py` |
| `LibraryPattern` dataclass | `dependencies/core.py:102-113` | `compute/dependencies/classification.py:85-113` | Delete from `core.py` |

### 5.2 Canonical Locations

After cleanup, the single source of truth for each:

| Item | Canonical Location |
|------|-------------------|
| Semantic role constants | `analytics/compute/semantic_roles/classification.py` |
| Semantic role dataclasses | `analytics/compute/semantic_roles/classification.py` |
| Dependency constants | `analytics/compute/dependencies/classification.py` |
| Dependency dataclasses | `analytics/compute/dependencies/` modules |

---

## 6. Stale Documentation

### 6.1 Outdated Docstrings

| File | Line | Current Text | Issue | Fix |
|------|------|--------------|-------|-----|
| `compute/functions/goids.py` | 4-6 | "extracted to support direct usage without the deprecated adapter layer" | Adapter layer is long gone | Remove mention of deprecated adapter |
| `compute/functions/typedness.py` | 6-7 | "re-exports and extends...for backward compatibility" | Check if compat is still needed | Update or remove compat statement |
| `parsing/compute.py` | 6-7 | "Use these functions instead of the deprecated `flush()` methods" | `flush()` methods don't exist | Update to reference `to_rows()` |
| `history/history_timeseries.py` | 360, 373, 457, 470 | "Unused but kept for API consistency" | Unused parameter cruft | Consider removing unused params |

### 6.2 Module-Level Documentation Updates

#### `analytics/compute/functions/goids.py`

**Current (Lines 1-7):**
```python
"""Function GOID types and loading utilities.

This module provides data types and utilities for working with function
global object identifiers (GOIDs). These types were originally in
analytics.adapters.functions and were extracted to support direct usage
without the deprecated adapter layer.
"""
```

**Recommended:**
```python
"""Function GOID types and loading utilities.

This module provides data types and utilities for working with function
global object identifiers (GOIDs), including loading from the database
and grouping by file path.
"""
```

---

## 7. Implementation Plan

### Phase 1: Immediate Safe Deletions (Day 1)

**Tasks:**
1. [ ] Delete `analytics/runtime/` directory
2. [ ] Delete `analytics/graphs/plugin_catalog.py`
3. [ ] Delete `analytics/graphs/contracts.py`
4. [ ] Run full test suite to verify

**Commands:**
```bash
rm -rf src/codeintel/analytics/runtime/
rm src/codeintel/analytics/graphs/plugin_catalog.py
rm src/codeintel/analytics/graphs/contracts.py

# Verify
uv run pytest -q
uv run ruff check --fix
uv run pyright --warnings --pythonversion=3.13
```

**Risk:** ✅ None - all items verified orphaned

---

### Phase 2: Semantic Roles Consolidation (Day 2-3)

**Tasks:**
1. [ ] Verify Hamilton native module at `build/hamilton/native/analytics/semantic_roles.py` provides all needed functionality
2. [ ] Update any remaining imports from `semantic_roles/core.py` to use `compute/semantic_roles/`
3. [ ] Delete `analytics/semantic_roles/core.py`
4. [ ] Update `analytics/semantic_roles/__init__.py` to re-export from compute layer
5. [ ] Run tests

**Verification Before Delete:**
```bash
# Check all imports
grep -r "from codeintel.analytics.semantic_roles" src/
grep -r "analytics.semantic_roles.core" src/
grep -r "compute_semantic_roles" src/
```

**Risk:** ⚠️ Medium - verify Hamilton native module coverage first

---

### Phase 3: Dependencies Consolidation (Day 4)

**Tasks:**
1. [ ] Remove duplicate constants from `dependencies/core.py`
2. [ ] Update imports to use `compute/dependencies/` modules
3. [ ] Run tests

**Changes to `dependencies/core.py`:**
```python
# REMOVE these lines (80-86, 79):
CALLSITE_MEDIUM_THRESHOLD = 10
SEVERITY_SCORES = {...}

# ADD import:
from codeintel.analytics.compute.dependencies.classification import (
    CALLSITE_MEDIUM_THRESHOLD,
    SEVERITY_SCORES,
)
```

**Risk:** ⚠️ Low - just import path changes

---

### Phase 4: Documentation Cleanup (Day 5)

**Tasks:**
1. [ ] Update stale docstrings in `compute/functions/goids.py`
2. [ ] Update stale docstrings in `compute/functions/typedness.py`
3. [ ] Update stale docstrings in `parsing/compute.py`
4. [ ] Review and possibly remove unused parameters in `history/history_timeseries.py`

**Risk:** ✅ None - documentation only

---

### Phase 5: Final Verification (Day 6)

**Tasks:**
1. [ ] Run full quality report
2. [ ] Run all tests
3. [ ] Update ANALYTICS_CLEANUP_PLAN.md with completion status
4. [ ] Create PR

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

---

## 8. Verification Checklist

### Pre-Deletion Verification

For each file marked for deletion:

- [ ] `grep -r "from codeintel.analytics.X" src/` returns no results
- [ ] `grep -r "import codeintel.analytics.X" src/` returns no results
- [ ] File is not referenced in `__all__` of any `__init__.py`
- [ ] File is not referenced in lazy loading `_LAZY_ATTRS` dicts
- [ ] No test files specifically test this module
- [ ] CI passes after deletion

### Post-Cleanup Verification

- [ ] `uv run pytest -q` passes
- [ ] `uv run ruff check` passes
- [ ] `uv run pyright --warnings --pythonversion=3.13` passes
- [ ] `uv run pyrefly check` passes
- [ ] Hamilton build targets still work:
  - [ ] `semantic_roles`
  - [ ] `dependencies`
  - [ ] `function_metrics`
  - [ ] `graph_metrics`

---

## Appendix A: File Inventory

### Files to DELETE

| File | Lines | Reason |
|------|-------|--------|
| `analytics/runtime/` | 0 | Empty directory |
| `analytics/graphs/plugin_catalog.py` | 268 | Orphaned, no imports |
| `analytics/graphs/contracts.py` | 415 | Orphaned, no imports |
| `analytics/semantic_roles/core.py` | 708 | Duplicate of compute layer + has Hamilton replacement |

**Total Deletable:** ~1,391 lines

### Files to MODIFY

| File | Changes |
|------|---------|
| `analytics/semantic_roles/__init__.py` | Update to re-export from compute layer |
| `analytics/dependencies/core.py` | Remove duplicate constants, import from compute |
| `analytics/compute/functions/goids.py` | Update docstring |
| `analytics/compute/functions/typedness.py` | Update docstring |
| `analytics/parsing/compute.py` | Update docstring |

---

## Appendix B: Import Dependency Graph

```
analytics/
├── compute/           # KEEP - pure computation layer
│   ├── dependencies/  # KEEP - canonical location
│   ├── functions/     # KEEP - canonical location
│   ├── graphs/        # KEEP - canonical location
│   └── semantic_roles/# KEEP - canonical location
├── dependencies/
│   └── core.py        # MODIFY - remove duplicates
├── graphs/
│   ├── contracts.py   # DELETE - orphaned
│   └── plugin_catalog.py # DELETE - orphaned
├── runtime/           # DELETE - empty
└── semantic_roles/
    └── core.py        # DELETE - replaced by Hamilton + compute layer
```

---

## Appendix C: Related Documents

- [Hamilton Consolidation Plan](../Hamilton_consolidation/Hamilton_consolidation_phase5.md)
- [Legacy Code Deprecation Plan](../Hamilton_consolidation/legacy_code_deprecation_plan.md)
- [Storage Decommissioning Plan](../Hamilton_consolidation/storage_decommissioning_plan.md)
- [Previous Analytics Cleanup Plan](../archive_improvement_plans/Cleanup/ANALYTICS_CLEANUP_PLAN.md)

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-16 | 1.0 | Initial document created from comprehensive analysis |

