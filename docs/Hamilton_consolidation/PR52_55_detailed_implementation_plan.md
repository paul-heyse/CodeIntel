# PR-52 through PR-55: Hamilton Consolidation Final Phase — Detailed Implementation Plan

> **Status**: Planning Document  
> **Created**: December 14, 2025  
> **Last Updated**: December 14, 2025  
> **Scope**: Complete Hamilton consolidation with aggressive cleanup and Option B schema migration  
> **Prerequisites**: PR-51 ✅ Complete (all deprecated functions removed, allowlist empty)

---

## Contextual Overview: Hamilton Consolidation Journey

This section provides context on the Hamilton consolidation initiative, summarizing the completed work from Phase 1 (PRs 46-50) and Phase 2 (PR-51), with deep dives into the foundational PR-50 and transformative PR-51.

### The North Star Architecture

The Hamilton consolidation initiative established these guiding principles:

1. **Only `codeintel/build` orchestrates execution** — Planning, run history, manifests, asset catalog, lineage, and all decisions about *what to compute* live in the build layer
2. **Domain packages don't own orchestration runtimes** — `analytics/`, `graphs/`, `ingestion/` are pure compute libraries returning `ibis.Table`, `pd.DataFrame`, or lightweight Python structures
3. **All DB writes are materialization** — Either via Hamilton materializers (preferred) or a single write API owned by build; critical for schema enforcement, asset catalog, and lineage correctness
4. **One registry to rule them all** — `UnifiedRegistry` is the single source of truth; parallel plugin registries deleted
5. **No backward compatibility layers** — Since the code is in design (not production), we delete transitional APIs and update imports everywhere

---

### Phase 1 Summary: Foundation (PRs 46-50)

Phase 1 established the foundational architecture that enabled the aggressive cleanup in Phase 2.

| PR | Title | Goal | Key Outcomes |
|----|-------|------|--------------|
| **PR-46** | Graph Runtime Relocation | Move `GraphRuntime` from `analytics.runtime` to `graphs.runtime` | Graph runtime became a first-class `graphs/` citizen; cross-domain usage clarified |
| **PR-47** | GraphProvider Relocation | Move `GraphProvider` from `analytics.resources` to `graphs.resources` | Analytics consumes graphs, doesn't define them; clean dependency direction |
| **PR-48** | Plugin Registry Removal | Delete `build/plugin_registry.py`, use `UnifiedRegistry` everywhere | Single source of truth for target→plugin resolution; eliminated parallel registries |
| **PR-49** | Compat Re-export Purge | Delete compatibility wrapper modules | Removed `graphs/catalog.py`, `ingestion/ports/storage.py` and other re-export shims |
| **PR-50** | Architecture Guardrails | Add tests enforcing Hamilton-first conventions | Scan tests forbid `gateway.ibis.write()` outside build/storage; allowlist mechanism |

---

### PR-50 Deep Dive: Architecture Guardrails

PR-50 established the enforcement mechanism that made PR-51's aggressive cleanup possible. Rather than relying on code review to catch architectural violations, PR-50 added automated tests that fail CI when conventions are violated.

#### Three Guardrail Tests

**1. `test_pr50_no_plugin_registry_imports()`**

Scans all source files to ensure no code imports from the deleted `codeintel.build.plugin_registry` module. This enforces that all plugin resolution goes through `UnifiedRegistry`.

```python
# Forbidden pattern (will fail CI):
from codeintel.build.plugin_registry import get_plugin_for_target

# Correct pattern:
from codeintel.build.unified_registry import get_unified_registry
registry = get_unified_registry()
plugin = registry.get_plugin(target_name)
```

**2. `test_pr50_no_analytics_runtime_imports()`**

Scans all source files to ensure no code imports from the relocated `codeintel.analytics.runtime` package. This enforces that graph runtime is accessed via `codeintel.graphs.runtime`.

```python
# Forbidden pattern (will fail CI):
from codeintel.analytics.runtime import GraphRuntime

# Correct pattern:
from codeintel.graphs.runtime import GraphRuntime
```

**3. `test_pr50_no_ibis_write_outside_build_allowlist()`**

The most impactful guardrail. Uses AST parsing to detect `.ibis.write(...)` calls and ensures they only exist in:
- `src/codeintel/build/` (Hamilton materializers)
- `src/codeintel/storage/` (low-level storage infrastructure)
- Files explicitly in `ALLOWLIST_IBIS_WRITE_FILES` (temporary backward compat)

```python
# Implementation detail: AST-based detection
def _contains_ibis_write_call(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "write":
                target = func.value
                if isinstance(target, ast.Attribute) and target.attr == "ibis":
                    return True
    return False
```

#### The Allowlist Mechanism

PR-50 introduced `ALLOWLIST_IBIS_WRITE_FILES` — a set of file paths temporarily permitted to contain direct DB writes. This enabled incremental migration:

```python
# Initial state (PR-50): 11 files allowlisted
ALLOWLIST_IBIS_WRITE_FILES = {
    "src/codeintel/analytics/cfg_dfg/materialize.py",
    "src/codeintel/analytics/data_models/core.py",
    "src/codeintel/analytics/dependencies/core.py",
    "src/codeintel/analytics/entrypoints/core.py",
    "src/codeintel/analytics/testing/graph_metrics.py",
    "src/codeintel/analytics/parsing/validation.py",
    "src/codeintel/analytics/compute/coverage/functions.py",
    "src/codeintel/analytics/compute/data_models/usage.py",
    "src/codeintel/analytics/profiles/writer_guard.py",
    "src/codeintel/analytics/functions/function_history.py",
    "src/codeintel/analytics/history/history_timeseries.py",
}

# Final state (post-PR-51): Empty
ALLOWLIST_IBIS_WRITE_FILES: set[str] = set()
```

---

### PR-51 Deep Dive: Eliminate DB Writes in Analytics

PR-51 was the largest and most transformative PR in the consolidation initiative. It migrated all 22 direct `gateway.ibis.write()` calls from 11 analytics modules to Hamilton native materializers, then removed all deprecated code in an aggressive decommissioning phase.

#### Scope: 11 Modules, 22 Tables

| # | Module | Tables | Complexity |
|---|--------|--------|------------|
| 1 | `analytics/cfg_dfg/materialize.py` | 6 | High |
| 2 | `analytics/data_models/core.py` | 3 | Medium |
| 3 | `analytics/dependencies/core.py` | 2 | Medium |
| 4 | `analytics/entrypoints/core.py` | 2 | Medium |
| 5 | `analytics/testing/graph_metrics.py` | 2 | Low |
| 6 | `analytics/parsing/validation.py` | 2 | Low |
| 7 | `analytics/compute/coverage/functions.py` | 1 | Low |
| 8 | `analytics/compute/data_models/usage.py` | 1 | Low |
| 9 | `analytics/profiles/writer_guard.py` | 1 | Low |
| 10 | `analytics/functions/function_history.py` | 1 | Low |
| 11 | `analytics/history/history_timeseries.py` | 1 | Low |

#### Three Migration Patterns Established

PR-51 discovered and documented three distinct migration patterns:

**1. Standard Target Replacement** (most common)
- Create new Hamilton native module in `build/hamilton/native/analytics/`
- Register with `native_module=` in `registrations.py`
- Full Hamilton lifecycle management (skip detection, timing, asset catalog)

```python
# build/hamilton/native/analytics/data_models.py
@tag(domain="analytics", target="data_models", node_type="materialize")
def t__data_models(env: BuildEnv, graph: TargetGraph, t__data_models__compute: DataModelsResult) -> TargetRunRecord:
    executor = NativeTargetExecutor.for_target(env, graph, "data_models")
    if executor.should_skip():
        return executor.skip()
    
    def compute() -> dict[str, int]:
        ctx = MaterializationContext(...)
        ref = materialize_rows(ctx, "analytics.data_models", result.model_rows, DATA_MODELS_COLS)
        return {"analytics.data_models": ref.row_count or 0}
    
    return executor.execute(compute)
```

**2. Utility Class Enhancement** (for shared utilities)
- Add `to_rows()` methods to existing reporter/utility classes
- Create materialization helper functions
- Deprecate `flush()` methods; consumer decides when to materialize
- Used for `parsing/validation.py` where reporters are shared across multiple consumers

```python
# analytics/parsing/validation.py
class FunctionValidationReporter:
    def to_rows(self) -> tuple[tuple[object, ...], ...]:
        """Return accumulated rows without writing."""
        return tuple(self._rows)
    
    # flush() method REMOVED in decommissioning

# Usage in consumers:
validation_rows = reporter.to_rows()
backend.bulk_insert("analytics.function_validation", validation_rows, columns=COLS)
```

**3. Pure Function Extraction** (simplest)
- Expose internal row-building logic as public `build_*_rows()` function
- No new Hamilton target registration needed
- Existing plugin/consumer calls new function + policy backend

```python
# analytics/compute/data_models/usage.py
def build_data_model_usage_rows(gateway, snapshot, module_map, ast_by_goid) -> list[tuple[object, ...]]:
    """Pure compute returning rows for materialization."""
    return _build_usage_rows(...)  # Internal logic now public

# Consumer pattern:
rows = build_data_model_usage_rows(gateway, snapshot, ...)
ref = materialize_rows(ctx, "analytics.data_model_usage", rows, DATA_MODEL_USAGE_COLS)
```

#### Decommissioning: The Aggressive Cleanup

After migration, PR-51 executed an aggressive decommissioning phase that merged the originally planned Phase 3 and Phase 4 into a single atomic operation. This was necessary because deprecated functions and plugin classes had tight coupling.

**Deprecated Functions Removed (17 total):**

| Module | Functions Removed |
|--------|-------------------|
| `cfg_dfg/materialize.py` | `compute_cfg_metrics()`, `compute_dfg_metrics()` |
| `data_models/core.py` | `compute_data_models()`, `_persist_models()` |
| `dependencies/core.py` | `build_external_dependency_calls()`, `build_external_dependencies()` |
| `entrypoints/core.py` | `build_entrypoints()` |
| `testing/graph_metrics.py` | `compute_test_graph_metrics()` |
| `parsing/validation.py` | `flush()` methods (both reporters) |
| `compute/coverage/functions.py` | `compute_coverage_functions()` |
| `compute/data_models/usage.py` | `compute_data_model_usage()` |
| `profiles/writer_guard.py` | `write_rows_with_registry_guard()`, `create_profile_writer()`, `WriterContext` |
| `functions/function_history.py` | `compute_function_history()` |
| `history/history_timeseries.py` | `compute_history_timeseries()`, `compute_history_timeseries_gateways()` |

**Plugin Files Deleted (4 files):**
- `build/plugins/analytics/functions/history.py` (FunctionHistoryPlugin)
- `build/plugins/analytics/coverage/functions.py` (CoverageFunctionsPlugin)
- `build/plugins/analytics/data_models/usage.py` (DataModelUsagePlugin)
- `build/plugins/analytics/history/timeseries.py` (HistoryTimeseriesPlugin)

**Plugin Directories Deleted (5 directories):**
- `build/plugins/analytics/cfg_dfg/`
- `build/plugins/analytics/data_models/`
- `build/plugins/analytics/dependencies/`
- `build/plugins/analytics/entrypoints/`
- `build/plugins/analytics/history/`

**Test Files Deleted (18 files + 6 directories):**
- 12 PR51 test files (`test_pr51_*.py`) that tested deprecated functions
- 6 empty test directories (`tests/analytics/dependencies/`, etc.)
- `tests/analytics/integration/` (sample_repo only used by deleted tests)

**Source Files Deleted:**
- `src/codeintel/analytics/compute/coverage/functions.py` (reduced to empty docstring)

**CLI Handler Refactored:**

The `cli/handlers/history.py` was refactored to replace deprecated `compute_history_timeseries_gateways()`:

```python
# Before (deprecated):
compute_history_timeseries_gateways(repo, commits, gateways=gateways, db_resolver=resolver)

# After:
rows = build_history_timeseries_rows(repo, commits, gateways=gateways, db_resolver=resolver)
backend = DuckDBPolicyBackend(target_gateway)
backend.delete_for_snapshot("analytics.history_timeseries", repo=repo, commit=target_commit)
backend.bulk_insert("analytics.history_timeseries", rows, columns=list(HISTORY_TIMESERIES_COLS))
```

#### PR-51 Final Metrics

| Metric | Before | After |
|--------|--------|-------|
| Direct writes outside build | 22 | **0** |
| Allowlisted files | 11 | **0** |
| Deprecated functions | 17 | **0** |
| Plugin classes (migrated targets) | 8 | **0** |
| Native analytics modules | 4 | **12** |
| Tables via Hamilton | ~30 | **52** |

#### Consumer Code Updates (Phase 2)

Before removing deprecated functions, PR-51 updated all consumer code:

| Consumer | Deprecated Function | Update |
|----------|---------------------|--------|
| `analytics/testing/profiles/rows.py` | `write_rows_with_registry_guard` | → `write_rows_via_policy_backend` |
| `analytics/functions/metrics.py` | `FunctionValidationReporter.flush()` | → `to_rows()` + policy backend |
| `graphs/validation/findings.py` | `GraphValidationReporter.flush()` | → `to_rows()` + policy backend |

---

### Current Codebase State (Ready for PR-52)

With PR-51 complete, the codebase has achieved:

1. **Zero direct DB writes outside build/storage** — Architecture guardrail passes with empty allowlist
2. **12 Hamilton native modules** — All analytics targets use native materialization
3. **Pure compute modules** — All deprecated orchestration code removed; modules contain only column definitions, dataclasses, and pure helper functions
4. **Clean plugin structure** — Only non-migrated plugins remain (`config_data_flow`, `coverage/test_edges`)
5. **Updated consumers** — All code using deprecated patterns has been updated

The groundwork is now in place for the final consolidation phase (PR-52 through PR-55).

---

## Executive Summary

This document provides exhaustive implementation details for the final Hamilton consolidation phase (PR-52 through PR-55). With PR-51 complete, the codebase is in an excellent position for aggressive cleanup. The deprecated functions and their direct DB writes have been removed, and Hamilton native modules now own all analytics materialization.

### Key Design Decisions

1. **PR-54 Schema Migration: Option B Selected** — Move `SCHEMA_REGISTRY` ownership to `build.hamilton.contracts`, establishing Hamilton as the single authority for data contracts
2. **Aggressive Breaking Changes Permitted** — Since the code is in design (not production), we can make breaking changes to reach the optimal final architecture faster
3. **No Legacy Shims** — Remove all backward compatibility layers; update imports everywhere rather than maintaining transitional APIs

### PR Overview

| PR | Description | Complexity | Dependencies |
|----|-------------|------------|--------------|
| **PR-52** | Delete Legacy Orchestrators | Low | PR-51 ✅ |
| **PR-53** | Consolidate Compute to Core | Medium | PR-52 |
| **PR-54** | Schema Validation → Hamilton (Option B) | High | PR-53 |
| **PR-55** | Final Sweep & Taxonomy Cleanup | Low | PR-54 |

---

## Current State Analysis (Post-PR-51)

### What PR-51 Accomplished

1. **Deprecated Functions Removed**: All 17 deprecated functions with `gateway.ibis.write()` calls deleted
2. **Allowlist Empty**: `ALLOWLIST_IBIS_WRITE_FILES` is now `set()`
3. **Plugin Files Deleted**: 4 plugin files + 5 empty plugin directories removed
4. **Test Files Cleaned**: 12 PR51 test files + 6 empty test directories removed
5. **Native Modules Authoritative**: 12 Hamilton native modules in `build/hamilton/native/analytics/`

### Remaining Core.py Files Analysis

| File | Lines | Contents After PR-51 | Orchestration? |
|------|-------|---------------------|----------------|
| `analytics/cfg_dfg/materialize.py` | 138 | Column definitions only | ❌ No |
| `analytics/data_models/core.py` | 882 | Column defs + AST helpers + dataclasses | ❌ No |
| `analytics/dependencies/core.py` | 620 | Column defs + detection helpers + dataclasses | ❌ No |
| `analytics/entrypoints/core.py` | 534 | Column defs + detection helpers + dataclasses | ❌ No |
| `analytics/semantic_roles/core.py` | ~500 | Column defs + classification helpers | ❌ No |

**Key Finding**: The "legacy orchestrator" pattern as originally envisioned in the Phase 2 plan **no longer exists**. PR-51 already removed the orchestration code. What remains are:
- Column definitions (constants)
- Pure dataclasses (data models)
- Pure helper functions (no I/O, no side effects)

### Compute Code Landscape

```
src/codeintel/
├── core/compute/
│   ├── centrality.py       # 423 lines - Canonical pure centrality functions
│   └── __init__.py
├── graphs/compute/
│   ├── metrics/
│   │   ├── bipartite.py    # Bipartite graph metrics
│   │   ├── cfg.py          # CFG-specific metrics
│   │   ├── community.py    # Community detection
│   │   ├── components.py   # Connected components
│   │   ├── coupling.py     # Module coupling
│   │   ├── dfg.py          # DFG-specific metrics
│   │   ├── paths.py        # Path analysis
│   │   ├── statistics.py   # Graph statistics
│   │   └── structural.py   # Structural hole metrics
│   ├── callgraph/          # Call graph construction
│   ├── cfg.py              # CFG construction
│   ├── dfg.py              # DFG construction
│   └── ...
└── analytics/compute/
    ├── graphs/
    │   ├── centrality.py   # Delegates to core/compute/centrality ✓
    │   ├── components.py   # Delegates to graphs/compute/metrics ✓
    │   ├── structural.py   # Delegates to graphs/compute/metrics ✓
    │   ├── cfg.py          # Analytics-specific CFG wrappers
    │   ├── dfg.py          # Analytics-specific DFG wrappers
    │   ├── conversions.py  # Graph format conversions
    │   ├── projections.py  # Bipartite projections
    │   └── types.py        # Analytics-specific types
    ├── row_builders/       # Row tuple construction
    ├── functions/          # Function-level metrics
    ├── evidence/           # Evidence collection
    └── ...
```

**Key Finding**: The compute consolidation is already partially complete:
- `analytics/compute/graphs/centrality.py` delegates to `core/compute/centrality.py` ✓
- `analytics/compute/graphs/components.py` delegates to `graphs/compute/metrics/` ✓
- `analytics/compute/graphs/structural.py` delegates to `graphs/compute/metrics/` ✓

The remaining question is: **Should `graphs/compute/metrics/` move to `core/compute/`?**

### Schema Registry Current Usage

**Location**: `src/codeintel/config/datasets/schema_registry.py`

**Usage by Package** (51 files total):

| Package | Files Using SCHEMA_REGISTRY | Impact of Moving |
|---------|----------------------------|------------------|
| `config/datasets/` | 12 | Internal - will move with registry |
| `build/` | 6 | Consumers - natural fit for ownership |
| `tests/` | 15 | Update imports |
| `cli/` | 2 | Update imports |
| `serving/` | 1 | Update imports |
| `docs/` | 4 | Update imports |
| `.hypothesis/` | 10 | Cached - will regenerate |

---

## PR-52: Delete Legacy Orchestrators

### Goal

Remove any remaining modules that combine:
1. Runtime/context construction
2. Compute function invocation
3. Database writes (now routed through Hamilton)
4. Validation

### Revised Scope (Post-PR-51 Analysis)

**Original Plan**: Delete orchestration code from core.py files and similar.

**Revised Reality**: The orchestration code was already removed in PR-51 when we deleted the deprecated functions. What remains is:

1. **Empty or Near-Empty Plugin Directories** — Verify all have been cleaned
2. **Redundant Module Files** — Any `.py` files that are now just docstrings
3. **Unused Import Paths** — Legacy re-export modules

### Task 1: Audit Remaining Plugin Structure

```bash
# Check what remains in build/plugins/analytics/
tree src/codeintel/build/plugins/analytics/
```

**Expected State** (verify):
```
build/plugins/analytics/
├── __init__.py              # Updated exports
├── config_data_flow/        # NOT migrated - keep
│   ├── __init__.py
│   └── compute.py
├── coverage/
│   ├── __init__.py          # Only test_edges
│   └── test_edges.py        # NOT migrated - keep
├── functions/
│   └── __init__.py          # Empty __all__
└── tests/
    └── __init__.py          # Empty __all__
```

**Action Items**:
- [ ] Delete `build/plugins/analytics/functions/` if `__init__.py` is empty and no other files
- [ ] Delete `build/plugins/analytics/tests/` if `__init__.py` is empty and no other files
- [ ] Verify `config_data_flow/` and `coverage/test_edges.py` are intentionally NOT migrated

### Task 2: Identify Any Remaining Orchestration Patterns

Search for any remaining patterns that violate "Hamilton owns materialization":

```python
# Patterns to search for:
# 1. Direct gateway.ibis.write() outside build/
# 2. Functions that both compute AND write
# 3. "Materialize" functions outside build/hamilton/
```

**Execution**:
```bash
# Already covered by PR-50 guardrails, but verify:
rg "gateway\.ibis\.write\(" src/codeintel --type py -l | grep -v "build/"

# Should return empty (allowlist is empty)
```

### Task 3: Remove Redundant Module Stubs

Check for any `.py` files that are now just module docstrings with no exports:

```bash
# Find modules with only docstrings (no functions/classes)
for f in $(find src/codeintel/analytics -name "*.py" -not -name "__init__.py"); do
  if ! grep -q "^def \|^class \|^[A-Z_]* = " "$f"; then
    echo "$f"
  fi
done
```

**Known Candidates** (from PR-51):
- None expected - all stub files were already deleted

### Task 4: Verify CLI Handler Completeness

The CLI handler `cli/handlers/history.py` was refactored in PR-51. Verify no other CLI handlers still call deprecated functions:

```bash
rg "compute_.*_deprecated\|build_.*_deprecated" src/codeintel/cli --type py
# Should return empty
```

### Tests for PR-52

**File**: `tests/build/hamilton/test_pr52_no_legacy_orchestrators.py`

```python
"""PR-52: Verify no legacy orchestrators remain outside build system."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src" / "codeintel"


def _iter_py_files() -> list[Path]:
    """Iterate all Python files excluding __pycache__."""
    return [p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in str(p)]


class TestNoOrchestratorsOutsideBuild:
    """Verify orchestration patterns only exist in build layer."""

    def test_no_direct_writes_outside_build(self) -> None:
        """Verify no gateway.ibis.write() calls outside build/."""
        # This is the same as PR-50 guardrail but kept for completeness
        allow_prefixes = ("src/codeintel/build/", "src/codeintel/storage/")
        bad: list[str] = []
        for p in _iter_py_files():
            rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            if rel.startswith(allow_prefixes):
                continue
            text = p.read_text(encoding="utf-8")
            if ".ibis.write(" in text or "gateway.ibis.write(" in text:
                bad.append(rel)
        assert not bad, f"Direct DB writes outside build:\n" + "\n".join(bad)

    def test_no_deprecated_function_calls_in_cli(self) -> None:
        """Verify CLI handlers don't call deprecated compute functions."""
        cli_root = SRC_ROOT / "cli"
        deprecated_patterns = [
            "compute_cfg_metrics(",
            "compute_dfg_metrics(",
            "compute_data_models(",
            "compute_function_history(",
            "compute_history_timeseries(",
            "compute_test_graph_metrics(",
            "build_entrypoints(",
            "build_external_dependency_calls(",
            "build_external_dependencies(",
        ]
        bad: list[tuple[str, str]] = []
        for p in cli_root.rglob("*.py"):
            if "__pycache__" in str(p):
                continue
            text = p.read_text(encoding="utf-8")
            for pattern in deprecated_patterns:
                if pattern in text:
                    bad.append((str(p.relative_to(REPO_ROOT)), pattern))
        assert not bad, f"Deprecated calls in CLI:\n" + "\n".join(f"{p}: {pat}" for p, pat in bad)

    def test_empty_plugin_directories_removed(self) -> None:
        """Verify empty plugin directories were cleaned up."""
        plugins_root = SRC_ROOT / "build" / "plugins" / "analytics"
        
        # These should NOT exist (deleted in PR-51)
        should_not_exist = [
            "cfg_dfg",
            "data_models",
            "dependencies",
            "entrypoints",
            "history",
        ]
        
        for name in should_not_exist:
            path = plugins_root / name
            assert not path.exists(), f"Empty plugin directory should be deleted: {path}"

    def test_non_migrated_plugins_still_exist(self) -> None:
        """Verify plugins intentionally not migrated still exist."""
        plugins_root = SRC_ROOT / "build" / "plugins" / "analytics"
        
        # These should exist (not part of PR-51 migration)
        should_exist = [
            ("config_data_flow", "compute.py"),
            ("coverage", "test_edges.py"),
        ]
        
        for dir_name, file_name in should_exist:
            path = plugins_root / dir_name / file_name
            assert path.exists(), f"Non-migrated plugin should exist: {path}"
```

### PR-52 Success Criteria

| Criterion | Verification |
|-----------|--------------|
| No gateway.ibis.write() outside build/ | PR-50 guardrail passes |
| No deprecated function calls in CLI | New test passes |
| Empty plugin directories removed | New test passes |
| Non-migrated plugins preserved | New test passes |
| All existing tests pass | `pytest -q` |
| Quality checks clean | ruff, pyright, pyrefly |

### Estimated Effort: Low (2-4 hours)

**Rationale**: Most orchestration cleanup was done in PR-51. This PR is primarily verification and cleanup of any stragglers.

---

## PR-53: Consolidate Compute Code into `codeintel.core.compute`

### Goal

Establish clear ownership boundaries for compute code:

1. **`core/compute/`** — Generic, reusable algorithms (no domain-specific knowledge)
2. **`graphs/compute/`** — Graph construction and domain-specific graph algorithms
3. **`analytics/compute/`** — Analytics-specific wrappers that add context and row building

### Current State Deep Analysis

#### What's in `core/compute/`

```python
# src/codeintel/core/compute/centrality.py (423 lines)
# Pure NetworkX-based centrality functions:
- CentralityMetrics (dataclass)
- compute_pagerank()
- compute_betweenness()
- compute_closeness()
- compute_harmonic_centrality()
- compute_eigenvector_centrality()
- compute_degree_centrality()
- compute_in_degree_centrality()
- compute_out_degree_centrality()
- compute_all_centralities()
- centrality_to_rows()  # Row conversion helper
```

#### What's in `graphs/compute/metrics/`

```python
# Structural metrics (structural.py):
- compute_clustering_coefficient()
- compute_triangles()
- compute_core_number()
- compute_constraint()  # Structural holes
- compute_effective_size()

# Component analysis (components.py):
- find_connected()
- find_strongly_connected()
- find_weakly_connected()
- topological_layers()

# Statistics (statistics.py):
- compute_diameter_estimate()
- compute_avg_shortest_path_length()
- compute_condensation_layer_count()

# Community detection (community.py):
- detect_communities_louvain()
- modularity_score()

# Path analysis (paths.py):
- count_simple_paths()

# Bipartite (bipartite.py):
- bipartite_degrees()
- project_weighted()

# CFG-specific (cfg.py) - Domain-specific, keep in graphs
# DFG-specific (dfg.py) - Domain-specific, keep in graphs
# Coupling (coupling.py) - Module coupling, domain-specific
```

#### What's in `analytics/compute/graphs/`

```python
# centrality.py (240 lines) - WRAPPER:
- CentralityComputations (dataclass for overrides)
- centrality_directed() - Adds GraphContext, sampling
- centrality_undirected() - Adds GraphContext, structural holes
- neighbor_stats() - Unique to analytics
# → Imports from core/compute/centrality ✓

# components.py (191 lines) - WRAPPER:
- component_metadata() - Adds analytics bundling
- component_ids_undirected()
- global_graph_stats()
# → Imports from graphs/compute/metrics/components ✓

# structural.py (118 lines) - WRAPPER:
- structural_metrics() - Bundles multiple metrics
- bounded_simple_path_count()
# → Imports from graphs/compute/metrics/structural ✓

# types.py (100+ lines) - ANALYTICS-SPECIFIC TYPES:
- CentralityBundle
- ComponentBundle
- GlobalGraphStats
- StructuralMetrics
- NeighborStats
# → Keep in analytics (domain-specific shapes)

# cfg.py, dfg.py - Analytics-specific CFG/DFG wrappers
# conversions.py - Graph format conversions
# projections.py - Bipartite projections
```

### Decision Point: What Should Move to `core/compute/`?

**Option A: Move `graphs/compute/metrics/` to `core/compute/`**

Pros:
- Single canonical location for all pure graph algorithms
- Clearer "core = pure algorithms" narrative
- Eliminates `graphs/compute/metrics/` vs `core/compute/` ambiguity

Cons:
- Large move (9 files)
- `graphs/` package becomes just construction, not metrics
- May feel over-centralized

**Option B: Keep `graphs/compute/metrics/` in `graphs/`, only centrality in `core/`**

Pros:
- Less churn
- `graphs/` retains complete graph-related functionality
- `core/` stays minimal

Cons:
- Two locations for "pure graph algorithms" (core vs graphs)
- Developers may be confused where to add new algorithms

**Option C (RECOMMENDED): Establish Clear Taxonomy**

```
core/compute/         # ONLY cross-domain utilities (centrality, generic stats)
graphs/compute/       # ALL graph-specific algorithms (metrics, construction)
analytics/compute/    # Analytics wrappers that add context/bundling
```

**Rationale**: 
- Centrality is genuinely cross-domain (used by analytics, graphs validation, potentially serving)
- Structural metrics, components, etc. are graph-domain-specific
- This avoids moving code that doesn't need to move

### PR-53 Implementation Plan (Option C)

#### Phase A: Audit Current Delegation

Verify all analytics/compute/graphs/ modules properly delegate:

```python
# Each of these should import from either core/compute or graphs/compute:
files_to_audit = [
    "analytics/compute/graphs/centrality.py",    # Delegates to core ✓
    "analytics/compute/graphs/components.py",    # Delegates to graphs ✓
    "analytics/compute/graphs/structural.py",    # Delegates to graphs ✓
    "analytics/compute/graphs/projections.py",   # Check delegation
    "analytics/compute/graphs/conversions.py",   # Check delegation
]
```

#### Phase B: Move Any Missing Generic Algorithms to Core

Based on audit, identify algorithms that should be in `core/compute/`:

**Candidates**:
- `compute_density()` (if not in core)
- `compute_node_degree_distribution()` (if not in core)

**NOT Candidates** (stay in graphs):
- All CFG/DFG-specific metrics
- Module coupling metrics
- Community detection (graph-domain-specific)
- Structural holes (graph-domain-specific)

#### Phase C: Ensure Consistent Import Patterns

All analytics/compute/ modules should follow this pattern:

```python
# analytics/compute/graphs/some_module.py

# Import pure compute from appropriate layer
from codeintel.core.compute.centrality import compute_pagerank  # Cross-domain
from codeintel.graphs.compute.metrics.structural import compute_clustering  # Graph-domain

# Define analytics-specific wrappers
def analytics_specific_wrapper(graph, ctx: GraphContext) -> AnalyticsBundle:
    """Add context, sampling, and bundle results."""
    raw_pagerank = compute_pagerank(graph)
    raw_clustering = compute_clustering(graph)
    return AnalyticsBundle(pagerank=raw_pagerank, clustering=raw_clustering)
```

#### Phase D: Add Architecture Test

Create test enforcing the import hierarchy:

```python
# tests/build/hamilton/test_pr53_compute_hierarchy.py

def test_analytics_compute_imports_from_core_or_graphs() -> None:
    """Verify analytics.compute imports from core.compute or graphs.compute."""
    analytics_compute = Path("src/codeintel/analytics/compute")
    
    # These modules should NOT import networkx directly for algorithms
    # They should delegate to core/compute or graphs/compute
    
    forbidden_in_analytics = [
        "nx.pagerank",
        "nx.betweenness_centrality",
        "nx.closeness_centrality",
        "nx.clustering",
        "nx.triangles",
    ]
    
    for py_file in analytics_compute.rglob("*.py"):
        text = py_file.read_text()
        for pattern in forbidden_in_analytics:
            if pattern in text:
                # Allow if it's for type hints only
                if f"TYPE_CHECKING" in text and pattern in text.split("TYPE_CHECKING")[1]:
                    continue
                raise AssertionError(
                    f"{py_file} calls {pattern} directly. "
                    f"Delegate to core/compute or graphs/compute instead."
                )
```

### File-by-File Changes for PR-53

| File | Action | Notes |
|------|--------|-------|
| `analytics/compute/graphs/projections.py` | Audit | Check if delegates to graphs/compute |
| `analytics/compute/graphs/conversions.py` | Audit | Check if delegates to graphs/compute |
| `core/compute/__init__.py` | Update exports | Ensure all public functions exported |
| `graphs/compute/metrics/__init__.py` | Update exports | Ensure all public functions exported |

### Tests for PR-53

**File**: `tests/build/hamilton/test_pr53_core_compute_canonical.py`

```python
"""PR-53: Verify compute code hierarchy and canonical locations."""

from __future__ import annotations

from pathlib import Path
import ast
import re

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src" / "codeintel"


class TestComputeHierarchy:
    """Verify compute code follows the established hierarchy."""

    def test_core_compute_has_centrality(self) -> None:
        """Verify centrality functions are in core.compute."""
        from codeintel.core.compute.centrality import (
            compute_pagerank,
            compute_betweenness,
            compute_closeness,
            compute_eigenvector_centrality,
            CentralityMetrics,
        )
        assert callable(compute_pagerank)
        assert callable(compute_betweenness)
        assert callable(compute_closeness)
        assert callable(compute_eigenvector_centrality)

    def test_analytics_centrality_delegates_to_core(self) -> None:
        """Verify analytics centrality imports from core."""
        centrality_file = SRC_ROOT / "analytics" / "compute" / "graphs" / "centrality.py"
        text = centrality_file.read_text()
        
        # Should import from core.compute.centrality
        assert "from codeintel.core.compute.centrality import" in text
        
        # Should NOT call nx centrality functions directly (except in type hints)
        nx_calls = re.findall(r"nx\.(pagerank|betweenness_centrality|closeness_centrality)\(", text)
        assert not nx_calls, f"Direct nx calls found: {nx_calls}"

    def test_analytics_components_delegates_to_graphs(self) -> None:
        """Verify analytics components imports from graphs.compute."""
        components_file = SRC_ROOT / "analytics" / "compute" / "graphs" / "components.py"
        text = components_file.read_text()
        
        # Should import from graphs.compute.metrics
        assert "from codeintel.graphs.compute.metrics" in text

    def test_analytics_structural_delegates_to_graphs(self) -> None:
        """Verify analytics structural imports from graphs.compute."""
        structural_file = SRC_ROOT / "analytics" / "compute" / "graphs" / "structural.py"
        text = structural_file.read_text()
        
        # Should import from graphs.compute.metrics
        assert "from codeintel.graphs.compute.metrics" in text

    def test_no_circular_imports(self) -> None:
        """Verify no circular imports in compute hierarchy."""
        # core should not import from analytics or graphs.compute
        core_compute = SRC_ROOT / "core" / "compute"
        for py_file in core_compute.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            text = py_file.read_text()
            assert "from codeintel.analytics" not in text, f"{py_file} imports analytics"
            assert "from codeintel.graphs.compute" not in text, f"{py_file} imports graphs.compute"


class TestComputeExports:
    """Verify compute modules export expected functions."""

    def test_core_compute_exports(self) -> None:
        """Verify core.compute exports all centrality functions."""
        from codeintel.core.compute import centrality
        
        expected = {
            "compute_pagerank",
            "compute_betweenness",
            "compute_closeness",
            "compute_harmonic_centrality",
            "compute_eigenvector_centrality",
            "compute_all_centralities",
            "CentralityMetrics",
        }
        
        actual = set(centrality.__all__)
        missing = expected - actual
        assert not missing, f"Missing exports: {missing}"

    def test_graphs_compute_metrics_exports(self) -> None:
        """Verify graphs.compute.metrics exports expected functions."""
        from codeintel.graphs.compute import metrics
        
        # Check structural exports
        from codeintel.graphs.compute.metrics import structural
        assert hasattr(structural, "compute_clustering_coefficient")
        assert hasattr(structural, "compute_triangles")
        
        # Check components exports
        from codeintel.graphs.compute.metrics import components
        assert hasattr(components, "find_connected")
        assert hasattr(components, "find_strongly_connected")
```

### PR-53 Success Criteria

| Criterion | Verification |
|-----------|--------------|
| analytics/compute delegates properly | Architecture test passes |
| core/compute has all centrality functions | Export test passes |
| No circular imports | Hierarchy test passes |
| No direct nx.* calls in analytics/compute | Pattern test passes |
| All existing tests pass | `pytest -q` |
| Quality checks clean | ruff, pyright, pyrefly |

### Estimated Effort: Medium (4-6 hours)

**Rationale**: Mostly audit work and adding architecture tests. Minimal code movement since delegation is already in place.

---

## PR-54: Schema Validation Consolidation (Option B — Hamilton-First)

### Goal

Establish Hamilton as the single authority for data contracts by moving `SCHEMA_REGISTRY` ownership to `build.hamilton.contracts`. This creates a clean separation:

- **Build Layer Owns**: Schema definitions, validation, enforcement
- **Domain Layers Use**: Import schemas for type hints and documentation only

### Why Option B (Hamilton-First)?

1. **Alignment with North Star**: "Only `codeintel/build` orchestrates execution" — schema validation is part of execution
2. **Contract Enforcement at Write Boundary**: Hamilton materializers validate; this puts schemas where enforcement happens
3. **Clean Dependency Direction**: `build → config → core` becomes `build ← (consumers) imports schemas`
4. **Future-Proofing**: Schema versioning, migration, and evolution are build concerns

### Migration Strategy

#### Phase 1: Create New Schema Home in Build

**New Location**: `src/codeintel/build/hamilton/contracts/schemas/`

```python
# src/codeintel/build/hamilton/contracts/schemas/__init__.py
"""Dataset schema definitions and registry.

This module is the authoritative source for all dataset schemas.
Schemas define:
- Column names and types (via Pandera)
- Validation rules and constraints
- Row type definitions for type checking

Examples
--------
>>> from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
>>> schema = SCHEMA_REGISTRY.require("analytics.function_metrics")
>>> schema.column_names()
('function_goid_h128', 'urn', 'repo', ...)
"""

from codeintel.build.hamilton.contracts.schemas.registry import (
    SCHEMA_REGISTRY,
    DatasetSchemaRegistry,
    get_schema,
)

__all__ = [
    "SCHEMA_REGISTRY",
    "DatasetSchemaRegistry",
    "get_schema",
]
```

#### Phase 2: Move Schema Infrastructure

**Files to Move**:

| From | To |
|------|-----|
| `config/datasets/schema_registry.py` | `build/hamilton/contracts/schemas/registry.py` |
| `config/datasets/schema.py` | `build/hamilton/contracts/schemas/schema.py` |
| `config/datasets/schema_builder.py` | `build/hamilton/contracts/schemas/builder.py` |
| `config/datasets/pandera_schemas.py` | `build/hamilton/contracts/schemas/pandera_schemas.py` |
| `config/datasets/constraints.py` | `build/hamilton/contracts/schemas/constraints.py` |

**Files to Keep in `config/datasets/`**:

| File | Reason |
|------|--------|
| `__init__.py` | Re-export for backward compat (temporary) |
| `export.py` | Dataset export utilities (not schema-related) |
| `lineage.py` | Lineage tracking (moves later or stays) |
| `introspection.py` | Move with schemas |
| `row_binding_factory.py` | Move with schemas |
| `row_migration.py` | Move with schemas |
| `validation.py` | Move with schemas |
| `dependency_inference.py` | Move with schemas |
| `plugin_constraints.py` | Move with schemas |

#### Phase 3: Create Backward Compatibility Layer (Temporary)

For a smooth migration, keep `config/datasets/` as a re-export layer initially:

```python
# src/codeintel/config/datasets/schema_registry.py (after move)
"""Backward compatibility re-exports for SCHEMA_REGISTRY.

.. deprecated::
    Import from codeintel.build.hamilton.contracts.schemas instead.
    This module will be removed in a future version.
"""

from __future__ import annotations

import warnings

from codeintel.build.hamilton.contracts.schemas import (
    SCHEMA_REGISTRY as _SCHEMA_REGISTRY,
    DatasetSchemaRegistry,
    get_schema,
)

warnings.warn(
    "Importing SCHEMA_REGISTRY from codeintel.config.datasets is deprecated. "
    "Use codeintel.build.hamilton.contracts.schemas instead.",
    DeprecationWarning,
    stacklevel=2,
)

SCHEMA_REGISTRY = _SCHEMA_REGISTRY

__all__ = [
    "SCHEMA_REGISTRY",
    "DatasetSchemaRegistry",
    "get_schema",
]
```

#### Phase 4: Update All Import Sites (Aggressive Approach)

Since we're in design mode, skip the deprecation period and update all imports immediately:

**Import Update Pattern**:

```python
# Before:
from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY

# After:
from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
```

**Files to Update** (by package):

**Build Package** (6 files):
```bash
rg "from codeintel\.config\.datasets" src/codeintel/build --type py -l
# Update all to use build.hamilton.contracts.schemas
```

**CLI Package** (2 files):
- `cli/handlers/ops.py`
- `cli/handlers/storage.py`

**Serving Package** (1 file):
- `serving/contracts/operation_contract_reflection.py`

**Tests** (15 files):
- All test files importing SCHEMA_REGISTRY

#### Phase 5: Delete Old Location (Aggressive Cleanup)

After all imports updated:

```bash
# Delete old schema files
rm src/codeintel/config/datasets/schema_registry.py
rm src/codeintel/config/datasets/schema.py
rm src/codeintel/config/datasets/schema_builder.py
rm src/codeintel/config/datasets/pandera_schemas.py
rm src/codeintel/config/datasets/constraints.py
rm src/codeintel/config/datasets/introspection.py
rm src/codeintel/config/datasets/row_binding_factory.py
rm src/codeintel/config/datasets/row_migration.py
rm src/codeintel/config/datasets/validation.py
rm src/codeintel/config/datasets/dependency_inference.py
rm src/codeintel/config/datasets/plugin_constraints.py
```

**Keep in `config/datasets/`**:
- `__init__.py` — Updated to export only non-schema utilities
- `export.py` — Dataset export utilities
- `lineage.py` — Lineage tracking (consider moving to build later)

### New Directory Structure

```
src/codeintel/build/hamilton/contracts/
├── __init__.py                    # Existing - updated exports
├── enforced_gateway.py            # Existing
├── enforcement_hook.py            # Existing
├── enforcement.py                 # Existing
├── pandera_hook.py               # Existing - updated imports
└── schemas/                       # NEW
    ├── __init__.py               # NEW - Main schema exports
    ├── registry.py               # Moved from config/datasets/
    ├── schema.py                 # Moved from config/datasets/
    ├── builder.py                # Moved from config/datasets/
    ├── pandera_schemas.py        # Moved from config/datasets/
    ├── constraints.py            # Moved from config/datasets/
    ├── introspection.py          # Moved from config/datasets/
    ├── row_binding_factory.py    # Moved from config/datasets/
    ├── row_migration.py          # Moved from config/datasets/
    ├── validation.py             # Moved from config/datasets/
    ├── dependency_inference.py   # Moved from config/datasets/
    └── plugin_constraints.py     # Moved from config/datasets/
```

### Circular Import Prevention

**Potential Issue**: `build/hamilton/contracts/schemas/registry.py` currently calls:
```python
from codeintel.build.unified_registry import get_unified_registry
```

**Solution**: This is fine — build → build is allowed. The key is ensuring:
1. `core` never imports from `build`
2. `analytics`, `graphs`, `serving` import schemas from `build`
3. `config` keeps only non-schema utilities

### Tests for PR-54

**File**: `tests/build/hamilton/test_pr54_schema_registry_in_build.py`

```python
"""PR-54: Verify SCHEMA_REGISTRY is owned by build.hamilton.contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src" / "codeintel"


class TestSchemaRegistryLocation:
    """Verify schema registry is in build.hamilton.contracts."""

    def test_schema_registry_importable_from_build(self) -> None:
        """Verify SCHEMA_REGISTRY imports from build.hamilton.contracts.schemas."""
        from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
        
        assert SCHEMA_REGISTRY is not None
        # Verify it has expected methods
        assert hasattr(SCHEMA_REGISTRY, "get")
        assert hasattr(SCHEMA_REGISTRY, "require")
        assert hasattr(SCHEMA_REGISTRY, "all")

    def test_old_config_path_does_not_export_schema_registry(self) -> None:
        """Verify config.datasets no longer exports SCHEMA_REGISTRY."""
        # This test passes only after aggressive cleanup
        import codeintel.config.datasets as datasets
        
        # SCHEMA_REGISTRY should NOT be in config.datasets anymore
        assert not hasattr(datasets, "SCHEMA_REGISTRY"), (
            "SCHEMA_REGISTRY should be removed from config.datasets; "
            "import from build.hamilton.contracts.schemas instead"
        )

    def test_no_schema_registry_imports_from_config(self) -> None:
        """Verify no code imports SCHEMA_REGISTRY from config.datasets."""
        bad: list[str] = []
        pattern = "from codeintel.config.datasets import.*SCHEMA_REGISTRY"
        pattern_alt = "from codeintel.config.datasets.schema_registry import"
        
        for py_file in SRC_ROOT.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            text = py_file.read_text()
            if pattern in text or pattern_alt in text:
                bad.append(str(py_file.relative_to(REPO_ROOT)))
        
        assert not bad, (
            f"Files still import SCHEMA_REGISTRY from config.datasets:\n"
            + "\n".join(bad)
        )


class TestSchemaRegistryUsability:
    """Verify schema registry works correctly from new location."""

    def test_can_get_function_metrics_schema(self) -> None:
        """Verify function_metrics schema is accessible."""
        from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
        
        schema = SCHEMA_REGISTRY.get("analytics.function_metrics")
        assert schema is not None
        assert "function_goid_h128" in schema.column_names()

    def test_can_require_nonexistent_raises(self) -> None:
        """Verify require() raises for unknown schemas."""
        from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
        
        with pytest.raises(KeyError):
            SCHEMA_REGISTRY.require("nonexistent.table")


class TestBuildLayerOwnership:
    """Verify build layer properly owns schema validation."""

    def test_pandera_hook_uses_build_schemas(self) -> None:
        """Verify pandera_hook imports from schemas subpackage."""
        pandera_hook = SRC_ROOT / "build" / "hamilton" / "contracts" / "pandera_hook.py"
        text = pandera_hook.read_text()
        
        # Should import from sibling schemas module
        assert (
            "from codeintel.build.hamilton.contracts.schemas import" in text
            or "from .schemas import" in text
        )

    def test_build_context_uses_build_schemas(self) -> None:
        """Verify build/context.py imports schemas from build.hamilton.contracts."""
        context_file = SRC_ROOT / "build" / "context.py"
        text = context_file.read_text()
        
        # If it uses SCHEMA_REGISTRY, it should import from build
        if "SCHEMA_REGISTRY" in text:
            assert "from codeintel.build.hamilton.contracts.schemas import" in text
```

### PR-54 Success Criteria

| Criterion | Verification |
|-----------|--------------|
| SCHEMA_REGISTRY importable from build.hamilton.contracts.schemas | Import test passes |
| No SCHEMA_REGISTRY in config.datasets | Removal test passes |
| No imports from config.datasets.schema_registry | Scan test passes |
| Pandera hook uses build schemas | Code check passes |
| Build context uses build schemas | Code check passes |
| All existing tests pass | `pytest -q` |
| Quality checks clean | ruff, pyright, pyrefly |

### Estimated Effort: High (8-12 hours)

**Rationale**: Moving 10+ files, updating 50+ import sites, ensuring no circular imports.

### Risk Mitigation

1. **Create schemas/ subpackage first** — Get new location working before touching old
2. **Update imports in dependency order** — Start with build/, then work outward
3. **Run tests after each major file move** — Catch issues early
4. **Keep backup of old files** until PR is merged

---

## PR-55: Final Sweep & Taxonomy Cleanup

### Goal

Polish the consolidated codebase:
1. Clean up snapshot manifest taxonomy
2. Audit public API exports
3. Remove any dead imports
4. Ensure consistent documentation

### Task 1: Snapshot Manifest Taxonomy Cleanup

**File**: `tests/build/hamilton/snapshots/manifest.yaml`

**Current Tags** (from manifest):
```yaml
# Tag taxonomy:
#   - PR tags: pr08, pr09, pr10, pr11, pr12, pr13, pr14, pr15, pr16, pr21, pr22, pr23
#   - Command tags: graph, plan, explain, history, status
#   - Format tags: json, dot, mermaid, text
#   - Scope tags: tiny, integration
#   - Mode tags: generated, phase0, native
```

**Actions**:
- [ ] Remove `phase0` tag if no longer used (superseded by `native`)
- [ ] Add PR tags for new work: `pr52`, `pr53`, `pr54`, `pr55`
- [ ] Ensure all cases have appropriate tags
- [ ] Remove any orphan snapshots (`.json` files not referenced in manifest)

### Task 2: Public API Audit

**Files to Audit**:

| File | Action |
|------|--------|
| `build/hamilton/__init__.py` | Update `__all__`, remove deprecated references |
| `build/__init__.py` | Update `__all__`, clean exports |
| `analytics/__init__.py` | Remove deprecated function exports |
| `analytics/compute/__init__.py` | Ensure clean exports |

**Example Cleanup for `analytics/__init__.py`**:

```python
# Before (if deprecated exports remain):
__all__ = [
    "compute_function_metrics",  # Still valid
    "compute_cfg_metrics",       # REMOVE - deleted in PR-51
    ...
]

# After:
__all__ = [
    "compute_function_metrics",
    # All deprecated compute_* functions removed
]
```

### Task 3: Dead Import Sweep

Run a comprehensive dead import check:

```bash
# Find potentially unused imports
uv run ruff check --select F401 src/codeintel/

# Find imports of deleted modules
rg "from codeintel\.analytics\.runtime" src/codeintel --type py -l
rg "from codeintel\.build\.plugin_registry" src/codeintel --type py -l
rg "from codeintel\.config\.datasets\.schema_registry" src/codeintel --type py -l  # After PR-54
```

### Task 4: Documentation Consistency

Ensure all module docstrings reflect the new architecture:

**Pattern for Deprecated Module Stubs**:
```python
"""Column definitions for <module>.

This module provides constants used by:
- ``codeintel.build.hamilton.native.analytics.<module>`` (Hamilton native node)
- ``codeintel.analytics.<domain>.compute`` (pure compute functions)

The deprecated ``compute_*`` functions have been removed.
Use the Hamilton build system for materialization.
"""
```

**Pattern for Compute Modules**:
```python
"""Pure compute functions for <domain>.

This module provides stateless computation functions that:
- Take inputs (gateway, snapshot, graphs, etc.)
- Return row tuples or dataclasses
- Perform NO database writes

Materialization is handled by Hamilton native modules in:
``codeintel.build.hamilton.native.analytics/``
"""
```

### Tests for PR-55

**File**: `tests/build/hamilton/test_pr55_final_sweep.py`

```python
"""PR-55: Final sweep verification tests."""

from __future__ import annotations

from pathlib import Path
import yaml

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src" / "codeintel"
SNAPSHOTS_DIR = REPO_ROOT / "tests" / "build" / "hamilton" / "snapshots"


class TestSnapshotManifestTaxonomy:
    """Verify snapshot manifest has valid taxonomy."""

    def test_manifest_tags_are_valid(self) -> None:
        """Verify all case tags conform to taxonomy."""
        manifest_path = SNAPSHOTS_DIR / "manifest.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        valid_pr_tags = {f"pr{i:02d}" for i in range(8, 56)}  # pr08 through pr55
        valid_command_tags = {"graph", "plan", "explain", "history", "status"}
        valid_format_tags = {"json", "dot", "mermaid", "text"}
        valid_scope_tags = {"tiny", "integration"}
        valid_mode_tags = {"generated", "native", "analytics", "graphs", "export"}
        
        all_valid = valid_pr_tags | valid_command_tags | valid_format_tags | valid_scope_tags | valid_mode_tags
        
        invalid_tags: list[tuple[str, str]] = []
        for case in manifest.get("cases", []):
            for tag in case.get("tags", []):
                if tag not in all_valid:
                    invalid_tags.append((case["name"], tag))
        
        assert not invalid_tags, f"Invalid tags found:\n" + "\n".join(f"{name}: {tag}" for name, tag in invalid_tags)

    def test_no_phase0_tags_remain(self) -> None:
        """Verify phase0 tag has been removed."""
        manifest_path = SNAPSHOTS_DIR / "manifest.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        phase0_cases = [
            case["name"] for case in manifest.get("cases", [])
            if "phase0" in case.get("tags", [])
        ]
        
        assert not phase0_cases, f"Cases still use phase0 tag:\n" + "\n".join(phase0_cases)

    def test_all_snapshots_referenced(self) -> None:
        """Verify all snapshot files are referenced in manifest."""
        manifest_path = SNAPSHOTS_DIR / "manifest.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        referenced = {case["snapshot"] for case in manifest.get("cases", []) if "snapshot" in case}
        
        actual_files = {p.name for p in SNAPSHOTS_DIR.glob("*.json")}
        actual_files.discard("manifest.yaml")  # Not a snapshot
        
        orphans = actual_files - referenced
        assert not orphans, f"Orphan snapshot files:\n" + "\n".join(sorted(orphans))


class TestPublicApiClean:
    """Verify public APIs are clean."""

    def test_analytics_no_deprecated_exports(self) -> None:
        """Verify analytics package doesn't export deprecated functions."""
        import codeintel.analytics as analytics
        
        deprecated = [
            "compute_cfg_metrics",
            "compute_dfg_metrics",
            "compute_data_models",
            "compute_function_history",
            "compute_history_timeseries",
            "compute_test_graph_metrics",
            "build_entrypoints",
            "build_external_dependencies",
        ]
        
        exported = []
        for name in deprecated:
            if hasattr(analytics, name):
                exported.append(name)
        
        assert not exported, f"Deprecated functions still exported:\n" + "\n".join(exported)

    def test_build_hamilton_no_legacy_references(self) -> None:
        """Verify build.hamilton doesn't reference legacy modes."""
        import codeintel.build.hamilton as hamilton
        
        # Should not have references to deleted modes
        assert not hasattr(hamilton, "phase0")
        assert not hasattr(hamilton, "legacy_mode")


class TestNoDeadImports:
    """Verify no dead imports remain."""

    def test_no_analytics_runtime_imports(self) -> None:
        """Verify no imports of deleted analytics.runtime."""
        bad: list[str] = []
        for py_file in SRC_ROOT.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            text = py_file.read_text()
            if "codeintel.analytics.runtime" in text:
                bad.append(str(py_file.relative_to(REPO_ROOT)))
        
        assert not bad, f"Files import deleted analytics.runtime:\n" + "\n".join(bad)

    def test_no_plugin_registry_imports(self) -> None:
        """Verify no imports of deleted plugin_registry."""
        bad: list[str] = []
        for py_file in SRC_ROOT.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            text = py_file.read_text()
            if "codeintel.build.plugin_registry" in text:
                bad.append(str(py_file.relative_to(REPO_ROOT)))
        
        assert not bad, f"Files import deleted plugin_registry:\n" + "\n".join(bad)
```

### PR-55 Success Criteria

| Criterion | Verification |
|-----------|--------------|
| Manifest tags conform to taxonomy | Taxonomy test passes |
| No phase0 tags remain | Tag removal test passes |
| No orphan snapshot files | Reference test passes |
| No deprecated exports in analytics | Export test passes |
| No dead imports | Import scan tests pass |
| All existing tests pass | `pytest -q` |
| Quality checks clean | ruff, pyright, pyrefly |

### Estimated Effort: Low (2-4 hours)

---

## Implementation Timeline

```
Week 1: PR-52 (Delete Legacy Orchestrators)
        ├── Day 1: Audit plugin structure, verify cleanup
        ├── Day 2: Add architecture tests
        └── Day 3: Final verification, merge

Week 2: PR-53 (Consolidate Compute)
        ├── Days 1-2: Audit compute delegation
        ├── Day 3: Add hierarchy tests
        └── Days 4-5: Any necessary code moves, merge

Week 3-4: PR-54 (Schema → Hamilton)
        ├── Days 1-2: Create schemas/ subpackage
        ├── Days 3-4: Move files
        ├── Days 5-6: Update all import sites
        ├── Day 7: Delete old location
        └── Days 8-9: Testing, merge

Week 5: PR-55 (Final Sweep)
        ├── Day 1: Manifest cleanup
        ├── Day 2: Public API audit
        ├── Day 3: Dead import sweep
        └── Day 4: Final verification, merge
```

---

## Success Metrics (Final State)

| Metric | Current (Post-PR-51) | After PR-55 |
|--------|---------------------|-------------|
| Direct writes outside build | 0 | 0 |
| Allowlisted files | 0 | 0 |
| Legacy orchestrator modules | 0 | 0 |
| Compute duplication | Minimal | None |
| Schema registry locations | 1 (config) | 1 (build) |
| Dead imports | Unknown | 0 |
| Orphan test files | Unknown | 0 |
| Public API deprecated exports | Unknown | 0 |

---

## Risk Assessment

### PR-52 Risks: Low
- Mostly verification work
- PR-51 already did the heavy lifting

### PR-53 Risks: Low-Medium
- Risk: Discovering unexpected duplication requiring larger moves
- Mitigation: Audit before committing to changes

### PR-54 Risks: Medium-High
- Risk: Circular imports when moving schema infrastructure
- Mitigation: Careful dependency analysis, incremental moves
- Risk: Missing import update sites
- Mitigation: Comprehensive grep + test runs after each change

### PR-55 Risks: Low
- Cleanup work with good test coverage
- No structural changes

---

## Appendix A: Quick Reference Commands

```bash
# Verify no direct writes outside build
rg "gateway\.ibis\.write\(" src/codeintel --type py -l | grep -v "build/"

# Find SCHEMA_REGISTRY imports
rg "SCHEMA_REGISTRY" src/codeintel --type py -l

# Check compute delegation
rg "from codeintel\.core\.compute" src/codeintel/analytics --type py

# Find dead imports
uv run ruff check --select F401 src/codeintel/

# Run architecture guardrails
uv run pytest tests/build/hamilton/test_pr50_architecture_guardrails.py -v

# Full quality check
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
```

---

## Appendix B: File Movement Checklist for PR-54

### Files to Move to `build/hamilton/contracts/schemas/`

| # | Source | Destination |
|---|--------|-------------|
| 1 | `config/datasets/schema_registry.py` | `schemas/registry.py` |
| 2 | `config/datasets/schema.py` | `schemas/schema.py` |
| 3 | `config/datasets/schema_builder.py` | `schemas/builder.py` |
| 4 | `config/datasets/pandera_schemas.py` | `schemas/pandera_schemas.py` |
| 5 | `config/datasets/constraints.py` | `schemas/constraints.py` |
| 6 | `config/datasets/introspection.py` | `schemas/introspection.py` |
| 7 | `config/datasets/row_binding_factory.py` | `schemas/row_binding_factory.py` |
| 8 | `config/datasets/row_migration.py` | `schemas/row_migration.py` |
| 9 | `config/datasets/validation.py` | `schemas/validation.py` |
| 10 | `config/datasets/dependency_inference.py` | `schemas/dependency_inference.py` |
| 11 | `config/datasets/plugin_constraints.py` | `schemas/plugin_constraints.py` |

### Files to Keep in `config/datasets/`

| File | Reason |
|------|--------|
| `__init__.py` | Update to export only remaining utilities |
| `export.py` | Dataset export (not schema-related) |
| `lineage.py` | Lineage tracking |

### New File to Create

| File | Purpose |
|------|---------|
| `build/hamilton/contracts/schemas/__init__.py` | Package exports |

---

## Appendix C: Import Update Sites for PR-54

### Build Package (Update to internal import)

```python
# src/codeintel/build/context.py
# Before:
from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY

# After:
from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
```

### CLI Package

```python
# src/codeintel/cli/handlers/ops.py
# src/codeintel/cli/handlers/storage.py
# Update same pattern
```

### Serving Package

```python
# src/codeintel/serving/contracts/operation_contract_reflection.py
# Update same pattern
```

### Test Files (15+)

```bash
# Find all test files needing updates
rg "from codeintel\.config\.datasets" tests/ --type py -l
```

---

**End of Document**
