# Integrated Cross-Package Decommissioning Plan

> **Generated:** 2025-12-16  
> **Packages:** `codeintel.core`, `codeintel.analytics`, `codeintel.graphs`, `codeintel.ingestion`  
> **Status:** Planning Phase  
> **Priority:** High - Holistic Architecture Alignment

## Executive Summary

This document provides an integrated view of cleanup and consolidation opportunities across the four core domain packages: `core`, `analytics`, `graphs`, and `ingestion`. By examining these packages holistically, we can identify cross-cutting opportunities that were not visible when analyzing each package in isolation.

**Key Findings:**
- **Clear layering is established**: `core` has no imports from analytics/graphs/ingestion ✅
- **Cross-package dependency exists**: `graphs` → `analytics` (validation import) - needs attention
- **Duplicate ParsedFunction/ParsedModule models**: `core.parsing` vs `graphs.ports.parsing`
- **Protocol consolidation opportunity**: Multiple similar protocols could be unified
- **Clean hexagonal architecture**: All packages follow proper port/adapter patterns
- **Validation framework consolidation**: Core provides base, packages extend properly

**Estimated Total Impact:**
- ~1,500-2,000 lines of dead/duplicate code can be removed across all packages
- 3-5 cross-package consolidation opportunities
- Improved architectural clarity and maintainability

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Package Dependency Analysis](#2-package-dependency-analysis)
3. [Cross-Package Consolidation Opportunities](#3-cross-package-consolidation-opportunities)
4. [Package-Specific Cleanup (Summary)](#4-package-specific-cleanup-summary)
5. [Holistic Implementation Plan](#5-holistic-implementation-plan)
6. [Best-in-Class Architecture Recommendations](#6-best-in-class-architecture-recommendations)
7. [Verification Strategy](#7-verification-strategy)

---

## 1. Architecture Overview

### Current Package Roles

```
┌─────────────────────────────────────────────────────────────────────┐
│                       HAMILTON BUILD LAYER                          │
│    (build/hamilton/native/*)  - Orchestration & Materialization     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ imports
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DOMAIN PACKAGES                             │
│                                                                     │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │
│  │   analytics   │  │    graphs     │  │   ingestion   │           │
│  │   (compute)   │  │   (compute)   │  │   (compute)   │           │
│  │               │  │   (engine)    │  │   (adapters)  │           │
│  │               │  │   (validation)│  │   (engine)    │           │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘           │
│          │                  │                  │                    │
│          └──────────────────┼──────────────────┘                    │
│                             │ all import                            │
│                             ▼                                       │
│              ┌─────────────────────────────┐                        │
│              │            core             │                        │
│              │   (protocols, types, errors)│                        │
│              │   (validation, resources)   │                        │
│              │   (parsing, schemas)        │                        │
│              └─────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Package Statistics

| Package | Python Files | Protocols | Adapters | Compute Modules |
|---------|-------------|-----------|----------|-----------------|
| `core` | 120 | 20+ | 0 | 1 |
| `analytics` | 70+ | 9 | 0 | 30+ |
| `graphs` | 35+ | 4 | 0 | 15+ |
| `ingestion` | 45+ | 6 | 5 | 10+ |

---

## 2. Package Dependency Analysis

### Import Graph (Verified)

```
core (0 external imports from sibling packages) ✅
  ↑
  ├── analytics (imports core.*)
  │     ├── core.schemas.generated_rows.analytics (15 files)
  │     ├── core.catalog (10+ files)
  │     ├── core.paths (8+ files)
  │     ├── core.validation (5+ files)
  │     └── core.parsing (5+ files)
  │
  ├── graphs (imports core.*, analytics.*)
  │     ├── core.validation (base classes)
  │     ├── core.resources (protocols)
  │     └── analytics.parsing.validation ⚠️ (cross-domain import)
  │
  └── ingestion (imports core.*)
        ├── core.ports.storage
        └── (no analytics or graphs imports) ✅

analytics → graphs (proper: analytics orchestrates graph operations)
  └── graphs.runtime.* (10+ imports for graph metric computation)

analytics → ingestion (proper: analytics uses ingestion infrastructure)
  └── ingestion.infrastructure.ast_utils (4 imports)
  └── ingestion.engine.infrastructure.ToolRunner (4 imports)
```

### Layering Violations to Address

| From | To | Import | Issue | Recommendation |
|------|-------|--------|-------|----------------|
| `graphs.validation.findings` | `analytics.parsing.validation` | `GRAPH_VALIDATION_COLS`, `GraphValidationReporter` | graphs should not import from analytics | Move `GraphValidationReporter` to `graphs.validation` or `core.validation` |

---

## 3. Cross-Package Consolidation Opportunities

### 3.1 Duplicate ParsedFunction/ParsedModule Models (HIGH PRIORITY)

**Issue:** Two separate definitions exist for parsed code models with different field sets.

**Location A:** `core/parsing/models.py` (147 lines)
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

**Location B:** `graphs/ports/parsing.py` (163 lines)
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

**Usage Analysis:**
- `core.parsing.ParsedFunction` - used by `analytics.parsing.*` (12+ files)
- `graphs.ports.parsing.ParsedFunction` - used by `graphs.compute.callgraph` (1 file)

**Recommendation:**
1. **Consolidate to `core.parsing.models`** as the canonical location
2. Add missing fields from `graphs.ports.parsing` to `core.parsing.models`
3. Create a factory/adapter to construct the simplified view for graphs when needed
4. Update `graphs.compute.callgraph.collection` to import from core

```python
# core/parsing/models.py - Unified ParsedFunction
@dataclass(frozen=True)
class ParsedFunction:
    # Core identification
    path: Path
    qualname: str
    function_goid_h128: int | None
    
    # Location
    span: SourceSpan
    
    # AST metadata
    ast: Any
    is_async: bool = False
    decorator_names: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    
    # Typing info
    docstring: str | None = None
    param_annotations: Mapping[str, Any] = field(default_factory=dict)
    return_annotation: Any | None = None
    param_any_flags: Mapping[str, bool] = field(default_factory=dict)
    return_is_any: bool = False
    
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

**Lines Saved:** ~100+ lines after consolidation

---

### 3.2 Validation Framework Layer Violation (HIGH PRIORITY)

**Issue:** `graphs.validation.findings` imports from `analytics.parsing.validation`, creating an incorrect dependency direction.

**Current Import:**
```python
# graphs/validation/findings.py
from codeintel.analytics.parsing.validation import (
    GRAPH_VALIDATION_COLS,
    GraphValidationReporter,
)
```

**Root Cause:** `GraphValidationReporter` was originally placed in analytics but is used by graphs.

**Recommendation:**
1. **Option A (Recommended):** Move `GraphValidationReporter` to `core.validation` as it's a shared infrastructure component
2. **Option B:** Move `GraphValidationReporter` to `graphs.validation` since graphs is the primary consumer
3. Update all imports accordingly

**Implementation (Option A):**
```python
# Move to: core/validation/reporters.py
class GraphValidationReporter:
    """Reporter for graph validation findings."""
    ...

GRAPH_VALIDATION_COLS: tuple[str, ...] = (...)
```

---

### 3.3 Storage Port Duplication (MEDIUM PRIORITY)

**Issue:** `ingestion/ports/storage.py` re-exports from `core.ports.storage` with additional protocol.

**Current:**
```python
# ingestion/ports/storage.py
from codeintel.core.ports.storage import BatchResult, MutableQueryResult, QueryResult, StoragePort

@runtime_checkable
class IngestStoragePort(Protocol):
    """Port protocol for persisting ingestion data."""
    ...
```

**Assessment:** This is acceptable - `IngestStoragePort` extends the core `StoragePort` with ingestion-specific methods. The re-exports provide convenient imports for ingestion code.

**Recommendation:** Keep current structure but update the docstring to clarify this is intentional domain-specific extension, not backward compatibility.

---

### 3.4 Validation Constants Consolidation (MEDIUM PRIORITY)

**Issue:** Graph validation constants are well-organized in `graphs.validation.findings` but analytics has similar patterns.

**Current State:**
```python
# graphs/validation/findings.py (well-organized)
SAMPLE_LIMIT = 5
SYMBOL_COMMUNITY_MIN = 2
CONFIG_KEY_MIN_THRESHOLD = 2
HUB_MIN_DEGREE_FLOOR = 10
HUB_DEGREE_RATIO = 0.1
CALL_SCC_MIN = 5
```

```python
# analytics/semantic_roles/core.py (duplicate)
ROLE_THRESHOLD = 0.35
SERVICE_FAN_IN_THRESHOLD = 5
SERVICE_FAN_OUT_THRESHOLD = 5
HELPER_LOC_THRESHOLD = 20
```

**Recommendation:** After completing analytics cleanup (removing `semantic_roles/core.py`), constants will be properly centralized in compute layers. No cross-package consolidation needed.

---

### 3.5 Protocol Naming Alignment (LOW PRIORITY)

**Issue:** Similar protocols have inconsistent naming patterns across packages.

| Package | Protocol | Naming Pattern |
|---------|----------|----------------|
| `core.resources` | `ResourceProvider[T_co]` | Verb + noun |
| `core.repository` | `RepositoryProtocol[T]` | Noun + Protocol |
| `core.catalog` | `CatalogProtocol` | Noun + Protocol |
| `graphs.engine` | `GraphEngine` | Noun (no Protocol suffix) |
| `ingestion.ports` | `IngestToolPort`, `IngestStoragePort` | Domain + Noun + Port |

**Recommendation:** Establish naming convention:
- **Ports** (boundary interfaces): `{Domain}{Noun}Port` (e.g., `IngestStoragePort`)
- **Engine/Service protocols**: `{Noun}Engine` or `{Noun}Service` (e.g., `GraphEngine`)
- **Generic protocols**: `{Noun}Protocol` (e.g., `RepositoryProtocol`)

This is documentation/convention work, not a code change priority.

---

## 4. Package-Specific Cleanup (Summary)

### 4.1 Analytics Package

See [ANALYTICS_DECOMMISSIONING_PLAN.md](./ANALYTICS_DECOMMISSIONING_PLAN.md) for details.

**Summary:**
- **Delete:** `runtime/` directory, `graphs/plugin_catalog.py`, `graphs/contracts.py`
- **Delete:** `semantic_roles/core.py` (duplicate of compute layer)
- **Consolidate:** Constants and dataclasses from `core.py` files to `compute/` modules
- **Estimated Lines Removed:** ~1,400

### 4.2 Graphs Package

See [GRAPHS_DECOMMISSIONING_PLAN.md](./GRAPHS_DECOMMISSIONING_PLAN.md) for details.

**Summary:**
- **Update:** 3 stale docstrings about "legacy function wrappers"
- **Review:** NetworkX version compatibility (may simplify if pinned to ≥3.0)
- **No orphaned modules** - package is clean
- **Estimated Lines Changed:** ~30-50

### 4.3 Ingestion Package

See [INGESTION_DECOMMISSIONING_PLAN.md](./INGESTION_DECOMMISSIONING_PLAN.md) for details.

**Summary:**
- **Update:** 1 docstring clarification in `ports/storage.py`
- **Review:** `ToolRunnerAdapter` usage (only in tests)
- **No dead code** - package is exemplary
- **Estimated Lines Changed:** ~10

### 4.4 Core Package

**Status:** Already cleaned in first pass. Minor documentation updates remain.

**Findings:**
- `core.ports.storage` has backward compatibility aliases (intentional, documented)
- `core.schemas.row_models` references "legacy RowBinding" (historical context, acceptable)
- `core.errors.taxonomy.STATUS_CODES` marked as "backward compatibility" (functional, keep)
- **No orphaned modules, no dead code**

**Recommendations:**
1. Keep backward compatibility aliases in `core.ports.storage` - they provide ingestion domain semantics
2. No immediate cleanup needed
3. Consider adding deprecation timeline to compatibility aliases if planning to remove them

---

## 5. Holistic Implementation Plan

### Phase 1: Cross-Package Critical Fixes (Days 1-2)

**Goal:** Fix layering violations and consolidate duplicate models.

**Tasks:**
1. [ ] Move `GraphValidationReporter` and `GRAPH_VALIDATION_COLS` from `analytics.parsing.validation` to `core.validation.reporters` (new module)
2. [ ] Update imports in `graphs.validation.findings` to use core
3. [ ] Update imports in `analytics.parsing.compute` to use core
4. [ ] Run full test suite

**Commands:**
```bash
# Create new module
touch src/codeintel/core/validation/reporters.py

# After moving code, verify no remaining cross-imports
grep -r "from codeintel.analytics" src/codeintel/graphs --include="*.py"
# Should return: No results

uv run pytest -q
```

### Phase 2: Consolidate Parsed Models (Days 3-4)

**Goal:** Unify ParsedFunction/ParsedModule to a single source of truth.

**Tasks:**
1. [ ] Extend `core.parsing.models.ParsedFunction` with fields from `graphs.ports.parsing`
2. [ ] Add compatibility properties (`name`, `start_line`, `end_line`)
3. [ ] Update `graphs.compute.callgraph.collection` to import from `core.parsing`
4. [ ] Deprecate `graphs.ports.parsing.ParsedFunction` and `ParsedModule`
5. [ ] Run tests

**Migration Example:**
```python
# Before (graphs/compute/callgraph/collection.py)
from codeintel.graphs.ports.parsing import ParsedModule

# After
from codeintel.core.parsing import ParsedModule
```

### Phase 3: Execute Package-Specific Cleanup (Days 5-8)

**Goal:** Complete the individual package cleanups in order of impact.

**Order:**
1. **Analytics** (highest impact): Delete orphaned modules, consolidate semantic_roles
2. **Graphs** (documentation): Update stale docstrings
3. **Ingestion** (minimal): Documentation update only
4. **Core** (none): No changes needed

See individual plan documents for detailed steps.

### Phase 4: Final Verification (Day 9)

**Tasks:**
1. [ ] Run full quality report
2. [ ] Run all tests including integration
3. [ ] Verify Hamilton build targets
4. [ ] Update documentation
5. [ ] Create PR

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
uv run pytest tests/build/ -q  # Hamilton targets
```

---

## 6. Best-in-Class Architecture Recommendations

### 6.1 Maintained Principles (✅ Already Implemented)

1. **Hexagonal Architecture**: All packages follow port/adapter patterns
2. **Layered Dependencies**: Core has no sibling imports (after fixing graphs→analytics)
3. **Pure Compute Separation**: `compute/` directories contain no I/O
4. **Protocol-Based Contracts**: Extensive use of `Protocol` for interfaces
5. **Hamilton Integration**: Build orchestration properly separated from domain logic

### 6.2 Future Architecture Improvements

**A. Resource Provider Unification**

Consider creating a single `ResourceBundle` pattern:

```python
# Future: Unified resource container
@dataclass
class AnalyticsResourceBundle:
    gateway: StorageGateway
    catalog: CatalogService
    graphs: GraphBundle
    
    @classmethod
    def from_context(cls, ctx: ExecutionContext) -> AnalyticsResourceBundle:
        """Factory from Hamilton execution context."""
        ...
```

**B. Validation Check Registry**

Move toward a unified check registry:

```python
# Future: core/validation/registry.py
class CheckRegistry:
    """Central registry for all validation checks."""
    
    def register(self, check: CheckProtocol, domains: tuple[str, ...]) -> None:
        """Register a check with its applicable domains."""
        ...
    
    def checks_for_domain(self, domain: str) -> tuple[CheckProtocol, ...]:
        """Get all checks applicable to a domain."""
        ...
```

**C. Error Taxonomy Usage**

The `core.errors.taxonomy` module is well-designed but underutilized. Consider:
- Making all package-specific exceptions inherit from taxonomy base classes
- Using `ErrorCode` instances consistently in exception metadata

---

## 7. Verification Strategy

### Pre-Implementation Checks

- [ ] All packages have passing tests
- [ ] No circular imports exist
- [ ] Type checking passes

### Per-Phase Verification

After each phase:

```bash
# Quick validation
uv run ruff check --fix
uv run pyright --warnings --pythonversion=3.13
uv run pytest -q

# Full validation
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
```

### Post-Implementation Verification

- [ ] `grep -r "from codeintel.analytics" src/codeintel/graphs` returns empty
- [ ] `grep -r "from codeintel.graphs" src/codeintel/analytics` shows only proper imports (to `graphs.runtime`)
- [ ] All Hamilton build targets succeed
- [ ] Documentation builds successfully
- [ ] No new linter errors introduced

---

## Appendix A: Combined File Inventory

### Files to DELETE (Across All Packages)

| Package | File | Lines | Reason |
|---------|------|-------|--------|
| analytics | `runtime/` | 0 | Empty directory |
| analytics | `graphs/plugin_catalog.py` | 268 | Orphaned |
| analytics | `graphs/contracts.py` | 415 | Orphaned |
| analytics | `semantic_roles/core.py` | 708 | Duplicate + Hamilton replacement |
| **Total** | | **~1,391** | |

### Files to MODIFY

| Package | File | Changes |
|---------|------|---------|
| core | `validation/__init__.py` | Export new reporters module |
| core | NEW: `validation/reporters.py` | Move GraphValidationReporter here |
| core | `parsing/models.py` | Add fields for graphs compatibility |
| analytics | `parsing/validation.py` | Remove GraphValidationReporter |
| analytics | `semantic_roles/__init__.py` | Update exports |
| analytics | `dependencies/core.py` | Remove duplicate constants |
| graphs | `validation/findings.py` | Import from core.validation |
| graphs | `validation/checks/*.py` | Update 3 docstrings |
| graphs | `ports/parsing.py` | Mark as deprecated, re-export from core |
| ingestion | `ports/storage.py` | Update docstring |

### Files to KEEP UNCHANGED

All `compute/` directories across packages - these are pure and properly designed.

---

## Appendix B: Cross-Package Import Map (Target State)

```
┌─────────────────────────────────────────────────────────────────────┐
│                            build/hamilton                           │
│                                                                     │
│    native/analytics/* ──────┬──────────────────────────────────┐    │
│    native/graphs/*    ──────┤                                  │    │
│    native/ingestion/* ──────┘                                  │    │
└───────────────────────────────┼──────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  analytics                graphs                 ingestion          │
│  ├── compute/*           ├── compute/*           ├── compute/*      │
│  ├── parsing/*           ├── engine/*            ├── adapters/*     │
│  └── ...                 ├── validation/*        ├── engine/*       │
│       │                  │      │                └── ports/*        │
│       │                  │      │                      │            │
│       │                  └──────┤                      │            │
│       │                         │                      │            │
│       ├─────────────────────────┴──────────────────────┘            │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                           core                              │    │
│  │  ├── validation/ (including reporters.py)                   │    │
│  │  ├── parsing/ (canonical ParsedFunction/Module)             │    │
│  │  ├── ports/                                                 │    │
│  │  ├── resources/                                             │    │
│  │  ├── schemas/                                               │    │
│  │  └── errors/                                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘

Arrows indicate allowed import directions.
No arrows from core to sibling packages.
No arrows from graphs to analytics (after cleanup).
```

---

## Appendix C: Related Documents

- [Analytics Decommissioning Plan](./ANALYTICS_DECOMMISSIONING_PLAN.md)
- [Graphs Decommissioning Plan](./GRAPHS_DECOMMISSIONING_PLAN.md)
- [Ingestion Decommissioning Plan](./INGESTION_DECOMMISSIONING_PLAN.md)
- [Hamilton Consolidation Plan](../Hamilton_consolidation/Hamilton_consolidation_phase5.md)
- [Storage Decommissioning Plan](../Hamilton_consolidation/storage_decommissioning_plan.md)

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-16 | 1.0 | Initial integrated document created |


