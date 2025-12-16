# Combined Post-Hamilton Decommissioning Plan

> **Generated:** 2025-12-16  
> **Scope:** `codeintel.core`, `codeintel.analytics`, `codeintel.graphs`, `codeintel.ingestion`  
> **Status:** Ready for Implementation  
> **Total Estimated Lines to Remove:** ~1,400  
> **Total Estimated Lines to Modify:** ~300

---

## Executive Summary

This document consolidates all cleanup and decommissioning opportunities identified across the four core domain packages following the Hamilton integration. It provides a single, actionable implementation plan with full context and rationale for each finding.

### Quick Stats by Package

| Package | Files to Delete | Files to Modify | Lines Removed | Priority |
|---------|-----------------|-----------------|---------------|----------|
| **analytics** | 4 | 5 | ~1,391 | 🔴 High |
| **graphs** | 0 | 4 | ~0 (doc only) | 🟡 Medium |
| **ingestion** | 0 | 1 | ~0 (doc only) | 🟢 Low |
| **core** | 0 | 2 | ~0 (new module) | 🔴 High |
| **Cross-Package** | 0 | 3 | ~100 | 🔴 High |

### Categories of Cleanup

1. **Dead Code Deletion** - Empty directories, orphaned modules (683 lines)
2. **Duplicate Code Removal** - Identical implementations in multiple places (708 lines)  
3. **Layering Violation Fixes** - Incorrect cross-package imports
4. **Documentation Updates** - Stale docstrings referencing removed code
5. **Model Consolidation** - Duplicate dataclass definitions

---

## Table of Contents

1. [Phase 1: Immediate Safe Deletions](#phase-1-immediate-safe-deletions)
2. [Phase 2: Cross-Package Layering Fix](#phase-2-cross-package-layering-fix)
3. [Phase 3: Semantic Roles Consolidation](#phase-3-semantic-roles-consolidation)
4. [Phase 4: Dependencies Consolidation](#phase-4-dependencies-consolidation)
5. [Phase 5: ParsedFunction Model Consolidation](#phase-5-parsedfunction-model-consolidation)
6. [Phase 6: Documentation Updates](#phase-6-documentation-updates)
7. [Final Verification](#final-verification)
8. [Appendix: Complete File Inventory](#appendix-complete-file-inventory)

---

## Phase 1: Immediate Safe Deletions

**Timeline:** Day 1  
**Risk Level:** ✅ None  
**Package:** analytics

These items have been verified as completely orphaned with zero imports anywhere in the codebase.

### 1.1 Delete Empty `runtime/` Directory

**Path:** `src/codeintel/analytics/runtime/`

**Rationale:**
- Directory contains only `__pycache__/` (bytecode cache)
- No Python modules exist in this directory
- No imports reference this location

**Verification Evidence:**
```bash
# Verify no imports exist
grep -r "analytics\.runtime\|from codeintel\.analytics import runtime" src/
# Expected: No matches

# Verify directory contents
ls -la src/codeintel/analytics/runtime/
# Expected: Only __pycache__/
```

**Implementation:**
```bash
rm -rf src/codeintel/analytics/runtime/
```

---

### 1.2 Delete Orphaned `graphs/plugin_catalog.py` (268 lines)

**Path:** `src/codeintel/analytics/graphs/plugin_catalog.py`

**Rationale:**
- Module exports 5 functions that are never imported anywhere
- Functionality superseded by `build/hamilton/` catalog generation
- Uses `codeintel.build.unified_registry` directly (build-layer concern)

**Module Contents (never imported):**
```python
def build_plugin_catalog() -> dict[str, Any]: ...
def render_plugin_catalog_markdown(catalog: dict[str, Any] | None = None) -> str: ...
def write_plugin_catalog(path: Path) -> None: ...
def write_plugin_catalog_markdown(path: Path, catalog: dict[str, Any] | None = None) -> None: ...
def write_plugin_catalog_html(path: Path, catalog: dict[str, Any] | None = None) -> None: ...
```

**Verification Evidence:**
```bash
# Search for any imports of this module
grep -r "from codeintel\.analytics\.graphs\.plugin_catalog" src/
grep -r "from codeintel\.analytics\.graphs import.*plugin_catalog" src/
# Expected: No matches

# Search for function names
grep -r "build_plugin_catalog\|render_plugin_catalog_markdown\|write_plugin_catalog" src/
# Expected: Only matches in the file itself
```

**Implementation:**
```bash
rm src/codeintel/analytics/graphs/plugin_catalog.py
```

---

### 1.3 Delete Orphaned `graphs/contracts.py` (415 lines)

**Path:** `src/codeintel/analytics/graphs/contracts.py`

**Rationale:**
- Contains contract checking infrastructure that was prototyped but never adopted
- Build layer has its own contract system at `build/hamilton/contracts/`
- Hard-coded `SAFE_TABLE_COLUMNS` dict indicates this was experimental
- All exports are never imported anywhere

**Module Contents (never imported):**
```python
@dataclass(frozen=True)
class PluginContractResult:
    name: str
    status: Literal["passed", "failed", "soft_failed"]
    message: str | None = None

def run_contract_checkers(...) -> tuple[PluginContractResult, ...]: ...
def assert_table_not_empty(...) -> PluginContractResult: ...
def assert_table_exists(...) -> PluginContractResult: ...
def assert_columns_present(...) -> PluginContractResult: ...
def table_not_empty_checker(...) -> ContractChecker: ...
# ... plus 4 more checker functions
```

**Verification Evidence:**
```bash
# Search for any imports
grep -r "from codeintel\.analytics\.graphs\.contracts" src/
grep -r "from codeintel\.analytics\.graphs import.*contracts" src/
# Expected: No matches

# Search for class/function names
grep -r "PluginContractResult\|run_contract_checkers\|table_not_empty_checker" src/
# Expected: Only matches in the file itself
```

**Implementation:**
```bash
rm src/codeintel/analytics/graphs/contracts.py
```

---

### Phase 1 Verification

After completing all deletions:

```bash
# Run full test suite
uv run pytest -q

# Run linter
uv run ruff check --fix

# Run type checker
uv run pyright --warnings --pythonversion=3.13

# Verify Hamilton targets still work
uv run pytest tests/build/ -q
```

**Expected Results:** All tests pass, no new errors introduced.

---

## Phase 2: Cross-Package Layering Fix

**Timeline:** Day 2  
**Risk Level:** ⚠️ Medium  
**Packages:** core, analytics, graphs

### 2.1 Fix graphs → analytics Import Violation

**Problem:**  
`graphs.validation.findings` imports from `analytics.parsing.validation`, which violates proper layering. The graphs package should not depend on analytics.

**Current Import (graphs/validation/findings.py, lines 14-17):**
```python
from codeintel.analytics.parsing.validation import (
    GRAPH_VALIDATION_COLS,
    GraphValidationReporter,
)
```

**Why This Is Wrong:**
- `core` is the foundational layer that all packages can import
- `analytics`, `graphs`, and `ingestion` should import from `core`, not from each other
- `graphs` importing from `analytics` creates a circular dependency risk

**Solution:** Move `GraphValidationReporter` and `GRAPH_VALIDATION_COLS` to `core.validation`.

### Implementation Steps

#### Step 2.1.1: Create `core/validation/reporters.py`

**New File:** `src/codeintel/core/validation/reporters.py`

```python
"""Validation reporters for structured finding collection.

This module provides reporter classes that collect validation findings
in a structured format suitable for persistence and analysis.

Classes
-------
FunctionValidationReporter
    Collects function-level validation findings.
GraphValidationReporter
    Collects graph-level validation findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Column definitions for validation tables
FUNCTION_VALIDATION_COLS: tuple[str, ...] = (
    "repo",
    "commit",
    "rel_path",
    "function_qualname",
    "function_goid_h128",
    "issue",
    "detail",
    "severity",
)

GRAPH_VALIDATION_COLS: tuple[str, ...] = (
    "repo",
    "commit",
    "graph_name",
    "entity_id",
    "issue",
    "detail",
    "severity",
    "rel_path",
    "metadata",
)


@dataclass
class FunctionValidationReporter:
    """Collect function validation findings for batch persistence.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.

    Examples
    --------
    >>> reporter = FunctionValidationReporter(repo="org/repo", commit="abc123")
    >>> reporter.record(
    ...     rel_path="src/main.py",
    ...     function_qualname="main",
    ...     function_goid_h128=12345,
    ...     issue="missing-docstring",
    ...     detail="Function has no docstring",
    ...     severity="warning",
    ... )
    >>> rows = reporter.to_rows()
    """

    repo: str
    commit: str
    _rows: list[tuple[object, ...]] = field(default_factory=list)

    def record(
        self,
        *,
        rel_path: str,
        function_qualname: str,
        function_goid_h128: int | None,
        issue: str,
        detail: str,
        severity: str = "info",
    ) -> None:
        """Record a function validation finding.

        Parameters
        ----------
        rel_path
            Relative path to the source file.
        function_qualname
            Fully qualified function name.
        function_goid_h128
            Function GOID hash (may be None).
        issue
            Issue identifier/code.
        detail
            Human-readable description of the issue.
        severity
            Severity level (info, warning, error).
        """
        self._rows.append((
            self.repo,
            self.commit,
            rel_path,
            function_qualname,
            function_goid_h128,
            issue,
            detail,
            severity,
        ))

    def to_rows(self) -> Sequence[tuple[object, ...]]:
        """Return collected rows for batch insertion.

        Returns
        -------
        Sequence[tuple[object, ...]]
            Rows in column order matching FUNCTION_VALIDATION_COLS.
        """
        return self._rows


@dataclass
class GraphValidationReporter:
    """Collect graph validation findings for batch persistence.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.

    Examples
    --------
    >>> reporter = GraphValidationReporter(repo="org/repo", commit="abc123")
    >>> reporter.record(
    ...     graph_name="import_graph",
    ...     entity_id="module.py",
    ...     issue="orphan-module",
    ...     detail="Module has no imports or exports",
    ... )
    >>> rows = reporter.to_rows()
    """

    repo: str
    commit: str
    _rows: list[tuple[object, ...]] = field(default_factory=list)

    def record(
        self,
        *,
        graph_name: str,
        entity_id: str,
        issue: str,
        detail: str,
        extras: dict[str, object] | None = None,
    ) -> None:
        """Record a graph validation finding.

        Parameters
        ----------
        graph_name
            Name of the graph being validated.
        entity_id
            Identifier of the entity with the finding.
        issue
            Issue identifier/code.
        detail
            Human-readable description.
        extras
            Optional additional metadata (severity, rel_path, metadata).
        """
        extras = extras or {}
        self._rows.append((
            self.repo,
            self.commit,
            graph_name,
            entity_id,
            issue,
            detail,
            extras.get("severity", "info"),
            extras.get("rel_path"),
            extras.get("metadata"),
        ))

    def to_rows(self) -> Sequence[tuple[object, ...]]:
        """Return collected rows for batch insertion.

        Returns
        -------
        Sequence[tuple[object, ...]]
            Rows in column order matching GRAPH_VALIDATION_COLS.
        """
        return self._rows


__all__ = [
    "FUNCTION_VALIDATION_COLS",
    "FunctionValidationReporter",
    "GRAPH_VALIDATION_COLS",
    "GraphValidationReporter",
]
```

#### Step 2.1.2: Update `core/validation/__init__.py`

Add exports for the new reporters module:

```python
# Add to existing exports:
from codeintel.core.validation.reporters import (
    FUNCTION_VALIDATION_COLS,
    FunctionValidationReporter,
    GRAPH_VALIDATION_COLS,
    GraphValidationReporter,
)

# Add to __all__:
__all__ = [
    # ... existing exports ...
    "FUNCTION_VALIDATION_COLS",
    "FunctionValidationReporter",
    "GRAPH_VALIDATION_COLS",
    "GraphValidationReporter",
]
```

#### Step 2.1.3: Update `graphs/validation/findings.py`

**Change imports from:**
```python
from codeintel.analytics.parsing.validation import (
    GRAPH_VALIDATION_COLS,
    GraphValidationReporter,
)
```

**To:**
```python
from codeintel.core.validation.reporters import (
    GRAPH_VALIDATION_COLS,
    GraphValidationReporter,
)
```

#### Step 2.1.4: Update `analytics/parsing/validation.py`

Keep the original implementations but have them import from core:

```python
# Re-export from core for backward compatibility
from codeintel.core.validation.reporters import (
    FUNCTION_VALIDATION_COLS,
    FunctionValidationReporter,
    GRAPH_VALIDATION_COLS,
    GraphValidationReporter,
)

__all__ = [
    "FUNCTION_VALIDATION_COLS",
    "FunctionValidationReporter",
    "GRAPH_VALIDATION_COLS",
    "GraphValidationReporter",
]
```

#### Step 2.1.5: Update `analytics/parsing/compute.py`

Update imports to use core:

```python
from codeintel.core.validation.reporters import (
    FUNCTION_VALIDATION_COLS,
    GRAPH_VALIDATION_COLS,
)
```

### Phase 2 Verification

```bash
# Verify no graphs → analytics imports remain
grep -r "from codeintel.analytics" src/codeintel/graphs --include="*.py"
# Expected: No matches

# Run tests
uv run pytest tests/graphs/ tests/analytics/ -q

# Run type checker
uv run pyright src/codeintel/graphs/ src/codeintel/analytics/ --pythonversion=3.13
```

---

## Phase 3: Semantic Roles Consolidation

**Timeline:** Days 3-4  
**Risk Level:** ⚠️ Medium  
**Package:** analytics

### 3.1 Problem Statement

Nearly identical implementations exist in two locations totaling ~1,000 lines:

| Location | Lines | Purpose |
|----------|-------|---------|
| `analytics/semantic_roles/core.py` | 708 | I/O + classification logic |
| `analytics/compute/semantic_roles/classification.py` | 454 | Pure classification logic |

### 3.2 Evidence of Duplication

The following elements are **identical** in both files:

| Element | `core.py` Lines | `classification.py` Lines |
|---------|-----------------|---------------------------|
| `ROLE_THRESHOLD = 0.35` | 30 | 24 |
| `SERVICE_FAN_IN_THRESHOLD = 5` | 31 | 25 |
| `SERVICE_FAN_OUT_THRESHOLD = 5` | 32 | 26 |
| `HELPER_LOC_THRESHOLD = 20` | 33 | 27 |
| `FunctionContext` dataclass | 36-99 | 30-88 |
| `RoleAccumulator` dataclass | 101-149 | 91-137 |
| `RoleArtifacts` dataclass | 152-163 | 140-150 |
| `ModuleRecord` dataclass | 165-171 | 153-158 |
| `classify_function_role()` | 455-466 | 161-187 |
| `_score_tests()` | 469-477 | 291-299 |
| `_score_api_handlers()` | 480-500 | 302-322 |
| `_score_cli_commands()` | 528-545 | 350-367 |
| `_score_repositories()` | 548-556 | 370-378 |
| `_score_services()` | 559-565 | 381-387 |
| `_score_validators()` | 568-578 | 390-400 |
| `_score_config_loaders()` | 581-585 | 403-407 |
| `_score_helpers()` | 588-594 | 410-416 |
| `_score_module_tags()` | 597-606 | 419-428 |
| `_score_module_hints()` | 609-612 | 431-434 |

### 3.3 Why `core.py` Should Be Deleted

1. **Violates separation of concerns**: Contains both I/O (DB writes) and pure logic
2. **Hamilton replacement exists**: `build/hamilton/native/analytics/semantic_roles.py` handles orchestration
3. **Pure version is canonical**: `compute/semantic_roles/classification.py` is designed for Hamilton integration
4. **Direct DB writes are anti-pattern**: Should use materializers exclusively

### 3.4 Implementation Steps

#### Step 3.4.1: Verify Hamilton Native Module Coverage

```bash
# Check what the Hamilton module imports
grep -n "from codeintel.analytics.semantic_roles" src/codeintel/build/hamilton/native/

# Expected: Shows import of compute_semantic_roles from semantic_roles module
```

#### Step 3.4.2: Update Hamilton Native Module

**File:** `src/codeintel/build/hamilton/native/analytics/semantic_roles.py`

Update to use pure functions from compute layer:

```python
# BEFORE
from codeintel.analytics.semantic_roles import compute_semantic_roles

# AFTER
from codeintel.analytics.compute.semantic_roles import (
    classify_function_role,
    classify_modules,
    FunctionContext,
    RoleArtifacts,
    ModuleRecord,
)
```

#### Step 3.4.3: Update `analytics/semantic_roles/__init__.py`

**Before:**
```python
from codeintel.analytics.semantic_roles.core import compute_semantic_roles
```

**After:**
```python
# Re-export pure functions from compute layer
from codeintel.analytics.compute.semantic_roles.classification import (
    classify_function_role,
    classify_modules,
    FunctionContext,
    ModuleRecord,
    RoleAccumulator,
    RoleArtifacts,
    HELPER_LOC_THRESHOLD,
    ROLE_THRESHOLD,
    SERVICE_FAN_IN_THRESHOLD,
    SERVICE_FAN_OUT_THRESHOLD,
)

__all__ = [
    "classify_function_role",
    "classify_modules",
    "FunctionContext",
    "ModuleRecord",
    "RoleAccumulator",
    "RoleArtifacts",
    "HELPER_LOC_THRESHOLD",
    "ROLE_THRESHOLD",
    "SERVICE_FAN_IN_THRESHOLD",
    "SERVICE_FAN_OUT_THRESHOLD",
]
```

#### Step 3.4.4: Delete `analytics/semantic_roles/core.py`

```bash
rm src/codeintel/analytics/semantic_roles/core.py
```

### Phase 3 Verification

```bash
# Verify no remaining imports from deleted module
grep -r "semantic_roles\.core" src/
grep -r "from codeintel\.analytics\.semantic_roles import compute_semantic_roles" src/
# Expected: No matches

# Run semantic roles tests
uv run pytest tests/analytics/ -k semantic_roles -q

# Verify Hamilton target still works
uv run pytest tests/build/ -k semantic -q
```

---

## Phase 4: Dependencies Consolidation

**Timeline:** Day 5  
**Risk Level:** ⚠️ Low  
**Package:** analytics

### 4.1 Problem Statement

The `analytics/dependencies/core.py` file contains duplicate constants and dataclasses that also exist in `analytics/compute/dependencies/`.

### 4.2 Duplicated Elements

| Element | `core.py` Lines | `compute/` Location |
|---------|-----------------|---------------------|
| `SEVERITY_SCORES` dict | 80-86 | `classification.py:18-24` |
| `CALLSITE_MEDIUM_THRESHOLD = 10` | 79 | `classification.py:26` |
| `DependencyModePattern` dataclass | 89-99 | `classification.py:29-57` |
| `LibraryPattern` dataclass | 102-113 | `classification.py:85-113` |
| `DependencyCall` dataclass | 116-129 | `detection.py` |

### 4.3 Implementation

**File:** `src/codeintel/analytics/dependencies/core.py`

**Remove these lines (approx. 79-129):**
```python
# DELETE: Duplicate constants
CALLSITE_MEDIUM_THRESHOLD = 10
SEVERITY_SCORES = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "info": 0,
}

# DELETE: Duplicate dataclasses
@dataclass(frozen=True)
class DependencyModePattern:
    ...

@dataclass(frozen=True)
class LibraryPattern:
    ...

@dataclass(frozen=True)
class DependencyCall:
    ...
```

**Add import at top:**
```python
from codeintel.analytics.compute.dependencies.classification import (
    CALLSITE_MEDIUM_THRESHOLD,
    SEVERITY_SCORES,
    DependencyModePattern,
    LibraryPattern,
)
from codeintel.analytics.compute.dependencies.detection import DependencyCall
```

### Phase 4 Verification

```bash
# Run dependency tests
uv run pytest tests/analytics/ -k dependencies -q

# Check for any broken imports
uv run pyright src/codeintel/analytics/dependencies/ --pythonversion=3.13
```

---

## Phase 5: ParsedFunction Model Consolidation

**Timeline:** Days 6-7  
**Risk Level:** ⚠️ Medium  
**Packages:** core, graphs

### 5.1 Problem Statement

Two different `ParsedFunction` definitions exist with different field sets:

**Location A: `core/parsing/models.py`**
```python
@dataclass(frozen=True)
class ParsedFunction:
    path: Path
    qualname: str
    function_goid_h128: int | None
    span: SourceSpan
    ast: Any
    docstring: str | None
    param_annotations: Mapping[str, Any]
    return_annotation: Any | None
    param_any_flags: Mapping[str, bool]
    return_is_any: bool
```

**Location B: `graphs/ports/parsing.py`**
```python
@dataclass(frozen=True)
class ParsedFunction:
    name: str
    qualname: str
    start_line: int
    end_line: int
    is_async: bool = False
    decorator_names: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
```

### 5.2 Usage Analysis

- `core.parsing.ParsedFunction` - used by 12+ files in `analytics.parsing.*`
- `graphs.ports.parsing.ParsedFunction` - used by 1 file (`graphs.compute.callgraph.collection`)

### 5.3 Solution: Extend Core Model

Add compatibility properties to `core/parsing/models.py`:

```python
@dataclass(frozen=True)
class ParsedFunction:
    # Existing fields...
    path: Path
    qualname: str
    function_goid_h128: int | None
    span: SourceSpan
    ast: Any
    docstring: str | None
    param_annotations: Mapping[str, Any]
    return_annotation: Any | None
    param_any_flags: Mapping[str, bool]
    return_is_any: bool
    
    # NEW: Fields for graphs compatibility
    is_async: bool = False
    decorator_names: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    
    # Compatibility properties
    @property
    def name(self) -> str:
        """Function name (for graphs compatibility)."""
        return self.qualname.rsplit(".", maxsplit=1)[-1]
    
    @property
    def start_line(self) -> int:
        """Start line (for graphs compatibility)."""
        return self.span.start_line
    
    @property
    def end_line(self) -> int:
        """End line (for graphs compatibility)."""
        return self.span.end_line
```

### 5.4 Update Graphs Import

**File:** `src/codeintel/graphs/compute/callgraph/collection.py`

**Change:**
```python
# BEFORE
from codeintel.graphs.ports.parsing import ParsedModule

# AFTER
from codeintel.core.parsing import ParsedModule
```

### 5.5 Deprecate graphs.ports.parsing

**File:** `src/codeintel/graphs/ports/parsing.py`

Add deprecation notice and re-export from core:

```python
"""Parsing data types for CST/AST operations.

.. deprecated::
    This module is deprecated. Import from ``codeintel.core.parsing`` instead.
    
This module re-exports parsing types from core for backward compatibility.
"""

from __future__ import annotations

import warnings

from codeintel.core.parsing import ParsedFunction, ParsedModule, SourceSpan

warnings.warn(
    "codeintel.graphs.ports.parsing is deprecated. "
    "Import from codeintel.core.parsing instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ParsedFunction",
    "ParsedModule",
    "SourceSpan",
]
```

### Phase 5 Verification

```bash
# Run graphs tests
uv run pytest tests/graphs/ -q

# Run analytics tests that use ParsedFunction
uv run pytest tests/analytics/ -k parsing -q

# Type check both packages
uv run pyright src/codeintel/graphs/ src/codeintel/core/parsing/ --pythonversion=3.13
```

---

## Phase 6: Documentation Updates

**Timeline:** Day 8  
**Risk Level:** ✅ None  
**Packages:** analytics, graphs, ingestion

### 6.1 Analytics Documentation Updates

#### 6.1.1 `analytics/compute/functions/goids.py` (lines 1-7)

**Current:**
```python
"""Function GOID types and loading utilities.

This module provides data types and utilities for working with function
global object identifiers (GOIDs). These types were originally in
analytics.adapters.functions and were extracted to support direct usage
without the deprecated adapter layer.
"""
```

**Updated:**
```python
"""Function GOID types and loading utilities.

This module provides data types and utilities for working with function
global object identifiers (GOIDs), including loading from the database
and grouping by file path.
"""
```

#### 6.1.2 `analytics/compute/functions/typedness.py` (lines 6-7)

**Current:**
```python
"""...re-exports and extends...for backward compatibility..."""
```

**Updated:**
Remove "backward compatibility" reference if no longer applicable.

#### 6.1.3 `analytics/parsing/compute.py` (lines 6-7)

**Current:**
```python
"""Use these functions instead of the deprecated `flush()` methods on the reporter
classes.
"""
```

**Updated:**
```python
"""Helper functions for converting validation reporters to row sequences.

These functions call `to_rows()` on reporter instances to extract data
for batch database insertion.
"""
```

### 6.2 Graphs Documentation Updates

#### 6.2.1 `graphs/validation/checks/anomaly.py` (lines 6-7)

**Current:**
```python
"""Anomaly detection validation checks.

Check classes implement CheckProtocol from core/validation; legacy
function wrappers are provided for backward compatibility.
"""
```

**Updated:**
```python
"""Anomaly detection validation checks.

Check classes implement CheckProtocol from core/validation.
"""
```

#### 6.2.2 `graphs/validation/checks/database.py` (lines 6-9)

**Current:**
```python
"""Database integrity validation checks.

This module contains validation checks that verify data integrity
by querying the database for inconsistencies.

Check classes implement CheckProtocol from core/validation; legacy
function wrappers are provided for backward compatibility.
"""
```

**Updated:**
```python
"""Database integrity validation checks.

This module contains validation checks that verify data integrity
by querying the database for inconsistencies.

Check classes implement CheckProtocol from core/validation.
"""
```

#### 6.2.3 `graphs/validation/checks/structure.py` (lines 6-9)

Same pattern as above - remove "legacy function wrappers" reference.

#### 6.2.4 `graphs/engine/views.py` (line 79)

**Add TODO:**
```python
    cycle_group :
        Cycle grouping id retained for backwards compatibility.
        TODO(cleanup): Review if cycle_group can be removed in favor of scc_id.
```

### 6.3 Ingestion Documentation Update

#### 6.3.1 `ingestion/ports/storage.py` (lines 3-4)

**Current:**
```python
"""Storage port protocol for ingestion data persistence.

This module re-exports unified storage types from ``codeintel.core.ports.storage``
with backward-compatible aliases for the ingestion naming convention.
"""
```

**Updated:**
```python
"""Storage port protocol for ingestion data persistence.

This module re-exports unified storage types from ``codeintel.core.ports.storage``
to provide domain-appropriate imports for ingestion code.
"""
```

---

## Final Verification

### Complete Quality Check

```bash
# Run full quality report
uv run python -m tools.quality_report --output build/quality-results/quality_report.json

# Run all tests
uv run pytest -q

# Run Hamilton build tests
uv run pytest tests/build/ -q
```

### Verify Layering

```bash
# No graphs → analytics imports
grep -r "from codeintel.analytics" src/codeintel/graphs --include="*.py"
# Expected: No matches

# Proper analytics → graphs imports (only to graphs.runtime)
grep -r "from codeintel.graphs" src/codeintel/analytics --include="*.py" | grep -v "graphs.runtime"
# Expected: No matches or only legitimate imports
```

### Hamilton Targets to Test

| Target | Test Command |
|--------|--------------|
| `semantic_roles` | `uv run pytest tests/build/ -k semantic -q` |
| `dependencies` | `uv run pytest tests/build/ -k dependencies -q` |
| `function_metrics` | `uv run pytest tests/build/ -k function_metrics -q` |
| `graph_metrics` | `uv run pytest tests/build/ -k graph_metrics -q` |
| `callgraph` | `uv run pytest tests/build/ -k callgraph -q` |
| `import_graph` | `uv run pytest tests/build/ -k import_graph -q` |

---

## Appendix: Complete File Inventory

### Files to DELETE

| Package | Path | Lines | Reason | Risk |
|---------|------|-------|--------|------|
| analytics | `runtime/` | 0 | Empty directory | ✅ None |
| analytics | `graphs/plugin_catalog.py` | 268 | Orphaned, zero imports | ✅ None |
| analytics | `graphs/contracts.py` | 415 | Orphaned, zero imports | ✅ None |
| analytics | `semantic_roles/core.py` | 708 | Duplicate + Hamilton replacement | ⚠️ Medium |

**Total Lines Deleted:** ~1,391

### Files to CREATE

| Package | Path | Purpose |
|---------|------|---------|
| core | `validation/reporters.py` | Move GraphValidationReporter from analytics |

### Files to MODIFY

| Package | Path | Changes |
|---------|------|---------|
| core | `validation/__init__.py` | Export new reporters module |
| core | `parsing/models.py` | Add fields for graphs compatibility |
| analytics | `parsing/validation.py` | Re-export from core |
| analytics | `parsing/compute.py` | Update docstring |
| analytics | `semantic_roles/__init__.py` | Re-export from compute layer |
| analytics | `dependencies/core.py` | Remove duplicate constants/classes |
| analytics | `compute/functions/goids.py` | Update docstring |
| analytics | `compute/functions/typedness.py` | Update docstring |
| graphs | `validation/findings.py` | Import from core.validation |
| graphs | `validation/checks/anomaly.py` | Update docstring |
| graphs | `validation/checks/database.py` | Update docstring |
| graphs | `validation/checks/structure.py` | Update docstring |
| graphs | `engine/views.py` | Add TODO comment |
| graphs | `ports/parsing.py` | Mark deprecated, re-export from core |
| graphs | `compute/callgraph/collection.py` | Update import |
| ingestion | `ports/storage.py` | Update docstring |

---

## Implementation Timeline

| Day | Phase | Package | Risk | Items |
|-----|-------|---------|------|-------|
| 1 | Phase 1 | analytics | ✅ None | Delete 3 orphaned items |
| 2 | Phase 2 | core, graphs | ⚠️ Medium | Fix layering violation |
| 3-4 | Phase 3 | analytics | ⚠️ Medium | Consolidate semantic_roles |
| 5 | Phase 4 | analytics | ⚠️ Low | Consolidate dependencies |
| 6-7 | Phase 5 | core, graphs | ⚠️ Medium | Consolidate ParsedFunction |
| 8 | Phase 6 | all | ✅ None | Documentation updates |
| 9 | Final | all | - | Verification & PR |

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-16 | 1.0 | Initial combined document created |


