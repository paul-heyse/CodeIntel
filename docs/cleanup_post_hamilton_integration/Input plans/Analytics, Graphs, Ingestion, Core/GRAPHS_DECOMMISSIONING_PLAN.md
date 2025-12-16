# Graphs Package Post-Hamilton Decommissioning Plan

> **Generated:** 2025-12-16  
> **Package:** `codeintel.graphs`  
> **Status:** Planning Phase  
> **Priority:** Medium - Technical Debt Reduction

## Executive Summary

The `graphs` package is well-structured following hexagonal architecture principles, but contains some stale documentation, unused compatibility code, and patterns that could be cleaned up. Unlike the `analytics` package, this package is more modern and doesn't have major duplication issues.

**Key Findings:**
- **3 stale docstrings** referencing non-existent legacy function wrappers
- **2 unused parameters** kept for protocol consistency
- **1 NetworkX backward compatibility shim** (may still be needed)
- **0 orphaned modules** - all modules are actively used
- **Clean layering** - compute functions properly separated from I/O

**Estimated Impact:**
- ~30-50 lines of documentation cleanup
- Improved code clarity
- Reduced confusion for contributors

---

## Table of Contents

1. [Package Architecture Overview](#1-package-architecture-overview)
2. [Stale Documentation](#2-stale-documentation)
3. [Unused Parameters (Protocol Compatibility)](#3-unused-parameters-protocol-compatibility)
4. [NetworkX Version Compatibility](#4-networkx-version-compatibility)
5. [Active Modules (No Changes Needed)](#5-active-modules-no-changes-needed)
6. [Cross-Package Opportunities](#6-cross-package-opportunities)
7. [Implementation Plan](#7-implementation-plan)
8. [Verification Checklist](#8-verification-checklist)

---

## 1. Package Architecture Overview

The `graphs` package follows a clean hexagonal architecture:

```
graphs/
├── __init__.py             # Public API exports
├── compute/                # Pure stateless computation
│   ├── callgraph/          # Call graph edge collection
│   ├── metrics/            # Graph metric computations
│   ├── cfg.py              # Control flow graph
│   ├── dfg.py              # Data flow graph
│   ├── goid.py             # GOID computation
│   ├── imports.py          # Import analysis
│   └── symbols.py          # Symbol use analysis
├── engine/                 # Graph engine protocol and implementation
│   ├── backend.py          # NetworkX backend configuration
│   ├── cache.py            # Graph caching
│   ├── factory.py          # Engine factory
│   ├── nx_engine.py        # NetworkX implementation
│   ├── protocol.py         # GraphEngine protocol
│   └── views.py            # DuckDB → NetworkX loaders
├── ports/                  # Data transfer objects
│   ├── engine.py           # GraphData DTO
│   └── parsing.py          # ParsedModule, ParsedFunction DTOs
├── resources/              # Dependency injection
│   ├── graph_provider.py   # GraphBundle provider
│   ├── graphs.py           # GraphResource
│   └── storage.py          # StorageResource
├── runtime/                # Runtime options and pooling
│   ├── context.py          # GraphContext, GraphMetricsOptions
│   └── runtime.py          # GraphRuntime, GraphRuntimeOptions
└── validation/             # Graph validation checks
    ├── base.py             # GraphCheckBase
    ├── context.py          # GraphValidationContext
    ├── findings.py         # Finding types, thresholds
    ├── runner.py           # Validation orchestration
    └── checks/             # Individual validation checks
        ├── anomaly.py      # Anomaly detection checks
        ├── database.py     # Database integrity checks
        └── structure.py    # Structure checks
```

**Assessment:** This is a well-designed package with clear separation of concerns. The cleanup opportunities are primarily documentation and minor code hygiene.

---

## 2. Stale Documentation

### 2.1 Validation Check Docstrings

**Issue:** Three validation check modules claim to provide "legacy function wrappers for backward compatibility" but no such wrappers exist in the code.

#### Location: `graphs/validation/checks/anomaly.py` (Lines 6-7)

**Current:**
```python
"""Anomaly detection validation checks.

Check classes implement CheckProtocol from core/validation; legacy
function wrappers are provided for backward compatibility.
"""
```

**Evidence of Non-Existence:**
```bash
# Search for any function wrappers in the file
grep -n "def warn_\|def check_\|# Legacy" src/codeintel/graphs/validation/checks/anomaly.py
# Result: Only class definitions, no legacy function wrappers
```

**Recommended Fix:**
```python
"""Anomaly detection validation checks.

Check classes implement CheckProtocol from core/validation.
"""
```

---

#### Location: `graphs/validation/checks/database.py` (Lines 6-7)

**Current:**
```python
"""Database integrity validation checks.

This module contains validation checks that verify data integrity
by querying the database for inconsistencies.

Check classes implement CheckProtocol from core/validation; legacy
function wrappers are provided for backward compatibility.
"""
```

**Recommended Fix:**
```python
"""Database integrity validation checks.

This module contains validation checks that verify data integrity
by querying the database for inconsistencies.

Check classes implement CheckProtocol from core/validation.
"""
```

---

#### Location: `graphs/validation/checks/structure.py` (Lines 6-7)

**Current:**
```python
"""Graph structure validation checks.

This module contains validation checks that analyze graph structure
for anomalies like cycles, hubs, and connectivity issues.

Check classes implement CheckProtocol from core/validation; legacy
function wrappers are provided for backward compatibility.
"""
```

**Recommended Fix:**
```python
"""Graph structure validation checks.

This module contains validation checks that analyze graph structure
for anomalies like cycles, hubs, and connectivity issues.

Check classes implement CheckProtocol from core/validation.
"""
```

---

### 2.2 Module-Level Documentation

#### Location: `graphs/engine/views.py` (Line 79)

**Current:**
```python
def module_attrs_from_row(
    ...
    cycle_group: int | Decimal | str | bytes | bytearray | None,
) -> tuple[str, dict[str, int]]:
    """
    ...
    cycle_group :
        Cycle grouping id retained for backwards compatibility.
    ...
    """
```

**Assessment:** This comment is accurate - `cycle_group` may be deprecated but is kept for compatibility. Consider adding a TODO for future removal or confirming it's still needed.

**Recommended Addition:**
```python
    cycle_group :
        Cycle grouping id retained for backwards compatibility.
        TODO(cleanup): Review if cycle_group can be removed in favor of scc_id.
```

---

## 3. Unused Parameters (Protocol Compatibility)

### 3.1 Protocol Consistency Parameters

Some functions have parameters that are unused but required for protocol/signature consistency. These are documented and intentional.

#### Location: `graphs/engine/views.py` (Lines 498-501)

```python
def load_symbol_function_graph(
    gateway: StorageGateway,
    _repo: str,
    _commit: str,
    *,
    use_gpu: bool = False,
) -> nx.Graph:
    """
    ...
    _repo : str
        Repository identifier (unused but required for protocol consistency).
    _commit : str
        Commit hash (unused but required for protocol consistency).
    ...
    """
```

**Assessment:** This is intentional - all graph loading functions share the same signature for consistency. The underscore prefix indicates intentional non-use. **No action needed.**

---

#### Location: `graphs/compute/metrics/coupling.py` (Lines 101-111)

```python
def compute_abstractness(
    _node: object,
    abstract_count: int,
    total_count: int,
) -> float:
    """Compute abstractness for a module.

    Parameters
    ----------
    _node
        Module identifier (unused, kept for signature compatibility).
    ...
    """
```

**Assessment:** This appears to be for a higher-order function pattern where the caller passes node ID but this function doesn't need it. The underscore prefix is correct. **No action needed.**

---

## 4. NetworkX Version Compatibility

### 4.1 NetworkX 3.x API Compatibility

**Location:** `graphs/engine/backend.py` (Lines 31-68)

```python
def _enable_nx_cugraph_backend() -> None:
    """
    Enable the nx-cugraph backend when available.

    Support both old API (set_default_backend) and NetworkX 3.x config API.
    ...
    """
    try:
        nx_cugraph = importlib.import_module("nx_cugraph")
    except ImportError as exc:
        ...

    # Try old API first
    set_backend = getattr(nx_cugraph, "set_default_backend", None)
    if set_backend is not None:
        set_backend()
        ...
        return

    # Try NetworkX 3.x config API
    try:
        nx = importlib.import_module("networkx")
        config = getattr(nx, "config", None)
        if config is not None and hasattr(config, "backend_priority"):
            config.backend_priority = ["cugraph"]
            ...
            return
    except (ImportError, AttributeError) as exc:
        ...

    # Fallback to environment variable
    os.environ.setdefault("NETWORKX_BACKEND_PRIORITY", "cugraph")
```

**Assessment:** This code handles multiple NetworkX versions (pre-3.0 and 3.x). Given that NetworkX 3.x is now mature, consider:

1. **Check minimum NetworkX version** in `pyproject.toml` - if it's ≥3.0, the old API branch can be removed
2. **Keep for now** if supporting older NetworkX versions is important

**Verification Command:**
```bash
grep -E "networkx.*[<>=]" pyproject.toml uv.lock
```

**Recommendation:** Review NetworkX version constraints. If pinned to ≥3.0, simplify the backend enablement code.

---

## 5. Active Modules (No Changes Needed)

The following modules were initially flagged as potentially unused but are actually actively used via Hamilton native modules:

### 5.1 `graphs/compute/dfg.py`

**Used by:** `build/hamilton/native/graphs/cfg_dfg.py`
```python
dfg_result = dfg_compute.build_dfg(goid, blocks, cfg_edges)
```

### 5.2 `graphs/compute/imports.py`

**Used by:** `build/hamilton/native/graphs/import_graph.py`
```python
result = imports_compute.analyze_imports(edges, modules)
```

### 5.3 `graphs/compute/symbols.py`

**Used by:** `build/hamilton/native/graphs/symbol_uses.py`
```python
edges = symbols_compute.build_use_edges(occurrences, def_map, module_by_path)
```

### 5.4 `graphs/compute/metrics/coupling.py`

**Used by:** `analytics/compute/graphs/` modules indirectly

**Assessment:** All compute modules are actively used. The initial unused detection missed Hamilton module imports.

---

## 6. Cross-Package Opportunities

### 6.1 Relationship with `analytics.graphs`

The `analytics.graphs` package has some overlap with `graphs.validation`:

| Analytics Module | Graphs Module | Relationship |
|------------------|---------------|--------------|
| `analytics/graphs/graph_stats.py` | `graphs/validation/findings.py` | Both define thresholds |
| `analytics/graphs/graph_metrics.py` | `graphs/compute/metrics/` | Analytics orchestrates, graphs computes |
| `analytics/graphs/contracts.py` | `graphs/validation/` | Duplicate contract checking (analytics orphaned) |

**Recommendation:** After deleting `analytics/graphs/contracts.py` (see Analytics cleanup plan), the validation framework will be cleanly in `graphs/validation/` only.

### 6.2 Shared Constants

The graphs package correctly centralizes validation constants:

**Canonical Location:** `graphs/validation/findings.py`
```python
SAMPLE_LIMIT = 5
CONFIG_KEY_MIN_THRESHOLD = 2
HUB_MIN_DEGREE_FLOOR = 10
HUB_DEGREE_RATIO = 0.1
CALL_SCC_MIN = 5
SYMBOL_COMMUNITY_MIN = 3
```

These are properly imported by `graphs/validation/checks/structure.py` and other modules.

**Assessment:** No duplication issues. The constant management is correct.

---

## 7. Implementation Plan

### Phase 1: Documentation Cleanup (Day 1)

**Tasks:**
1. [ ] Update docstring in `graphs/validation/checks/anomaly.py`
2. [ ] Update docstring in `graphs/validation/checks/database.py`
3. [ ] Update docstring in `graphs/validation/checks/structure.py`
4. [ ] Add TODO comment to `graphs/engine/views.py` for `cycle_group`
5. [ ] Run quality checks

**Changes:**

```bash
# File: src/codeintel/graphs/validation/checks/anomaly.py
# Change lines 6-7 from:
#   Check classes implement CheckProtocol from core/validation; legacy
#   function wrappers are provided for backward compatibility.
# To:
#   Check classes implement CheckProtocol from core/validation.

# File: src/codeintel/graphs/validation/checks/database.py
# Same change as above

# File: src/codeintel/graphs/validation/checks/structure.py
# Same change as above
```

**Verification:**
```bash
uv run ruff check src/codeintel/graphs/
uv run pytest tests/graphs/ -q
```

**Risk:** ✅ None - documentation only

---

### Phase 2: NetworkX Version Review (Day 2)

**Tasks:**
1. [ ] Check NetworkX version constraint in `pyproject.toml`
2. [ ] If ≥3.0, simplify `graphs/engine/backend.py`
3. [ ] Run GPU backend tests (if available)

**Decision Point:**
- If `networkx >= "3.0"` → Remove old API branch
- If `networkx` supports older versions → Keep compatibility code

**Risk:** ⚠️ Low - only if changing backend code

---

### Phase 3: Cross-Package Coordination (After Analytics Cleanup)

**Prerequisite:** Complete Analytics cleanup plan Phase 1 (delete orphaned modules)

**Tasks:**
1. [ ] Verify `analytics/graphs/contracts.py` is deleted
2. [ ] Confirm no references to deleted analytics modules from graphs
3. [ ] Run cross-package integration tests

---

## 8. Verification Checklist

### Pre-Change Verification

- [ ] All tests pass: `uv run pytest tests/graphs/ -q`
- [ ] Type checking passes: `uv run pyright src/codeintel/graphs/`
- [ ] Lint passes: `uv run ruff check src/codeintel/graphs/`

### Post-Change Verification

- [ ] All tests still pass
- [ ] No new type errors
- [ ] No new lint errors
- [ ] Graph validation works: Test with a sample repository

### Integration Verification

After both Analytics and Graphs cleanup:
- [ ] Hamilton build targets work:
  - [ ] `callgraph`
  - [ ] `import_graph`
  - [ ] `cfg_dfg`
  - [ ] `symbol_uses`
  - [ ] `graph_metrics`

---

## Appendix A: File Inventory

### Files to MODIFY

| File | Changes |
|------|---------|
| `graphs/validation/checks/anomaly.py` | Update docstring (lines 6-7) |
| `graphs/validation/checks/database.py` | Update docstring (lines 6-7) |
| `graphs/validation/checks/structure.py` | Update docstring (lines 6-7) |
| `graphs/engine/views.py` | Add TODO for cycle_group (line 79) |

### Files to DELETE

None - the graphs package is clean.

### Files to KEEP UNCHANGED

| File | Reason |
|------|--------|
| `graphs/compute/dfg.py` | Used by Hamilton |
| `graphs/compute/imports.py` | Used by Hamilton |
| `graphs/compute/symbols.py` | Used by Hamilton |
| `graphs/compute/metrics/coupling.py` | Used by analytics compute |
| `graphs/engine/backend.py` | May still need NX version compat |

---

## Appendix B: Module Usage Map

```
graphs/compute/dfg.py
└── Used by: build/hamilton/native/graphs/cfg_dfg.py

graphs/compute/imports.py
└── Used by: build/hamilton/native/graphs/import_graph.py

graphs/compute/symbols.py
└── Used by: build/hamilton/native/graphs/symbol_uses.py

graphs/compute/metrics/*
└── Used by: analytics/compute/graphs/* (proper layering)

graphs/validation/*
└── Used by: build/hamilton/native/graphs/graph_validation.py (presumably)
```

---

## Appendix C: Related Documents

- [Analytics Decommissioning Plan](./ANALYTICS_DECOMMISSIONING_PLAN.md)
- [Hamilton Consolidation Plan](../Hamilton_consolidation/Hamilton_consolidation_phase5.md)
- [Storage Decommissioning Plan](../Hamilton_consolidation/storage_decommissioning_plan.md)

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-16 | 1.0 | Initial document created from comprehensive analysis |


