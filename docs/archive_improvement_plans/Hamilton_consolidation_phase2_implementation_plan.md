# Hamilton Consolidation Phase 2 — Implementation Plan

> **Status**: Planning Document  
> **Created**: December 14, 2025  
> **Scope**: PRs 51–55 (remaining consolidation work after Phase 1)

This document outlines the implementation plan for completing the Hamilton consolidation initiative. Phase 1 (PRs 46–50) established the foundational architecture. Phase 2 focuses on migrating remaining direct writes, eliminating legacy orchestrators, and consolidating compute code.

---

## Executive Summary

### Completed in Phase 1
| PR | Description | Status |
|----|-------------|--------|
| PR-46 | Graph Runtime Relocation | ✅ Complete |
| PR-47 | GraphProvider Relocation | ✅ Complete |
| PR-48 | Plugin Registry Removal | ✅ Complete |
| PR-49 | Compat Re-export Purge | ✅ Complete |
| PR-50 | Architecture Guardrails | ✅ Tests Exist (allowlist active) |

### Remaining in Phase 2
| PR | Description | Effort | Priority |
|----|-------------|--------|----------|
| PR-51 | Eliminate DB Writes in Analytics | High | P0 |
| PR-52 | Delete Legacy Orchestrators | Medium | P1 |
| PR-53 | Consolidate Compute to Core | Medium | P1 |
| PR-54 | Schema Validation Consolidation | Low | P2 |
| PR-55 | Final Sweep & Taxonomy Cleanup | Low | P2 |

---

## Current State Analysis

### Direct Write Violations (PR-50 Allowlist)

The architecture guardrail test (`test_pr50_no_ibis_write_outside_build_allowlist`) currently allows 11 files with 22 direct `gateway.ibis.write()` calls:

| File | Writes | Domain | Migration Complexity |
|------|--------|--------|---------------------|
| `analytics/cfg_dfg/materialize.py` | 6 | CFG/DFG | High |
| `analytics/data_models/core.py` | 3 | Data Models | Medium |
| `analytics/parsing/validation.py` | 2 | Parsing | Low |
| `analytics/testing/graph_metrics.py` | 2 | Testing | Low |
| `analytics/dependencies/core.py` | 2 | Dependencies | Medium |
| `analytics/entrypoints/core.py` | 2 | Entrypoints | Medium |
| `analytics/compute/coverage/functions.py` | 1 | Coverage | Low |
| `analytics/compute/data_models/usage.py` | 1 | Data Models | Low |
| `analytics/profiles/writer_guard.py` | 1 | Profiles | Low |
| `analytics/functions/function_history.py` | 1 | Functions | Low |
| `analytics/history/history_timeseries.py` | 1 | History | Low |

### Compute Code Duplication

Current state shows duplication between:
- `src/codeintel/core/compute/centrality.py` — Pure centrality functions
- `src/codeintel/analytics/compute/graphs/centrality.py` — Analytics-specific wrappers

Additional compute code in `analytics/compute/` that may belong in `core/`:
- Graph structural metrics (`components.py`, `structural.py`)
- Pure transformations (`conversions.py`, `projections.py`)

---

## PR-51 — Eliminate DB Writes in Analytics/Graphs/Ingestion

### Goal

Convert all non-build modules to **pure compute** that returns expressions/DataFrames, with persistence happening exclusively through Hamilton materializers in `build/`.

### Implementation Strategy

#### Batch 1: Low-Complexity Migrations (Week 1)

Target files with single writes and clear boundaries:

1. **`analytics/compute/coverage/functions.py`**
   - [ ] Extract write logic into return-expression pattern
   - [ ] Create native node in `build/hamilton/native/analytics/coverage_functions.py`
   - [ ] Update `coverage_functions` target to use native materializer

2. **`analytics/compute/data_models/usage.py`**
   - [ ] Convert to pure compute returning `ibis.Table`
   - [ ] Wire through existing `data_models` build target

3. **`analytics/profiles/writer_guard.py`**
   - [ ] Analyze guard pattern — may need to become build-layer validation
   - [ ] Move write to materializer or delete if redundant

4. **`analytics/functions/function_history.py`**
   - [ ] Convert to expression-returning function
   - [ ] Integrate with `history_timeseries` target

5. **`analytics/history/history_timeseries.py`**
   - [ ] Pure compute for timeseries aggregation
   - [ ] Materialize via `history_timeseries` target

#### Batch 2: Medium-Complexity Migrations (Week 2)

Target files with multiple writes or dependencies:

6. **`analytics/dependencies/core.py`** (2 writes)
   - [ ] Map each write to a distinct output table
   - [ ] Create/update native nodes for dependency outputs

7. **`analytics/entrypoints/core.py`** (2 writes)
   - [ ] Analyze entrypoint detection flow
   - [ ] Wire through `entrypoints` build target

8. **`analytics/data_models/core.py`** (3 writes)
   - [ ] Identify distinct tables written
   - [ ] Create separate compute functions per table
   - [ ] Materialize via data model targets

9. **`analytics/parsing/validation.py`** (2 writes)
   - [ ] Determine if validation writes are runtime or build-time
   - [ ] Move to appropriate materializer

10. **`analytics/testing/graph_metrics.py`** (2 writes)
    - [ ] Convert test metric writes to pure compute
    - [ ] Materialize via `test_graph_metrics` target

#### Batch 3: High-Complexity Migrations (Week 3)

11. **`analytics/cfg_dfg/materialize.py`** (6 writes)
    - [ ] Audit all 6 write operations
    - [ ] Design multi-output native node structure
    - [ ] Create `build/hamilton/native/analytics/cfg_dfg.py`
    - [ ] Handle dependencies between CFG/DFG outputs

### Canonical Migration Pattern

**Before (domain writes):**
```python
# analytics/some_module/core.py
def compute_and_write(gateway: StorageGateway, snapshot: SnapshotRef) -> None:
    expr = compute_expression(...)
    gateway.ibis.write("analytics.some_table", expr)
```

**After (domain computes, build materializes):**
```python
# analytics/some_module/compute.py
def some_table_expr(inputs: SomeInputs) -> ir.Table:
    """Pure compute returning Ibis expression."""
    return compute_expression(...)

# build/hamilton/native/analytics/some_module.py
def some_table(
    ctx: TargetExecutionContext,
    some_table_expr: ir.Table,
) -> DatasetRef:
    """Materialize the computed expression."""
    return ctx.materialize_table("analytics.some_table", some_table_expr)
```

### Tests Checklist

- [ ] `tests/build/hamilton/test_pr51_write_boundary_shrinks.py`
  - [ ] Assert PR-50 allowlist is empty (or reduced to specific count)
  - [ ] Parametrized test for each migrated module
- [ ] Integration tests for each new native node
- [ ] Regression tests ensuring table schemas unchanged

### Success Criteria

- PR-50 allowlist reduced to **0 files** (or documented exceptions)
- All analytics compute modules are side-effect free
- Hamilton native materializers own all persistence

---

## PR-52 — Delete Legacy Orchestrators

### Goal

Remove modules that orchestrate computation + persistence outside the Hamilton build system.

### Identification Criteria

A module is a "legacy orchestrator" if it:
1. Builds a runtime or context
2. Calls compute functions
3. Validates results
4. Writes to database
5. Is **not** invoked by the Hamilton build system

### Candidate Modules for Review

| Module | Pattern | Action |
|--------|---------|--------|
| `analytics/cfg_dfg/materialize.py` | Orchestrates CFG/DFG pipeline | Migrate writes (PR-51), then delete orchestration |
| `analytics/data_models/core.py` | Orchestrates data model builds | Migrate, keep pure compute |
| `analytics/dependencies/core.py` | Orchestrates dependency detection | Migrate, keep detection logic |
| `analytics/entrypoints/core.py` | Orchestrates entrypoint analysis | Migrate, keep analysis logic |

### Tasks

- [ ] Audit each candidate module for orchestration patterns
- [ ] For each orchestrator:
  - [ ] Identify pure compute functions worth keeping
  - [ ] Extract to `compute/` subpackage if not already there
  - [ ] Delete orchestration wrapper after PR-51 migration
- [ ] Update all import sites to use Hamilton build targets

### Tests Checklist

- [ ] `tests/build/hamilton/test_pr52_no_legacy_orchestrators_imported.py`
  - [ ] Scan test forbidding imports of retired modules
  - [ ] Verify build targets provide equivalent functionality

---

## PR-53 — Consolidate Compute Code into `codeintel.core.compute`

### Goal

Establish `core/compute/` as the canonical location for pure, reusable graph/metric algorithms. Domain packages (`analytics/`, `graphs/`) call core compute, not duplicate it.

### Current Duplication Analysis

| Algorithm | Core Location | Analytics Location | Action |
|-----------|---------------|-------------------|--------|
| PageRank | `core/compute/centrality.py` | `analytics/compute/graphs/centrality.py` | Consolidate to core |
| Betweenness | `core/compute/centrality.py` | `analytics/compute/graphs/centrality.py` | Consolidate to core |
| Closeness | `core/compute/centrality.py` | `analytics/compute/graphs/centrality.py` | Consolidate to core |
| Components | — | `analytics/compute/graphs/components.py` | Move to core |
| Structural | — | `analytics/compute/graphs/structural.py` | Move to core |

### Migration Plan

#### Phase A: Centrality Consolidation

1. [ ] Audit `analytics/compute/graphs/centrality.py`
   - Identify any analytics-specific logic
   - Document differences from `core/compute/centrality.py`

2. [ ] Merge or redirect:
   - [ ] If identical: Delete analytics version, update imports
   - [ ] If different: Keep analytics as thin wrapper calling core

#### Phase B: Move Generic Graph Algorithms to Core

1. [ ] `analytics/compute/graphs/components.py` → `core/compute/components.py`
2. [ ] `analytics/compute/graphs/structural.py` → `core/compute/structural.py`
3. [ ] `analytics/compute/graphs/conversions.py` → `core/compute/conversions.py` (if generic)

#### Phase C: Keep Domain-Specific Code in Analytics

The following should remain in `analytics/compute/`:
- `graphs/cfg.py` — CFG-specific analysis
- `graphs/dfg.py` — DFG-specific analysis
- `graphs/types.py` — Analytics-specific type definitions
- `row_builders/` — Analytics row construction

### New Core Compute Structure

```
src/codeintel/core/compute/
├── __init__.py
├── centrality.py      # PageRank, betweenness, closeness, etc.
├── components.py      # Connected components, SCCs
├── structural.py      # Density, diameter, clustering
├── conversions.py     # Graph format conversions
└── projections.py     # Bipartite projections
```

### Tests Checklist

- [ ] `tests/build/hamilton/test_pr53_core_compute_is_canonical.py`
  - [ ] Import checks for moved modules
  - [ ] Verify old paths raise `ModuleNotFoundError`
- [ ] Unit tests for each core compute function
- [ ] Verify analytics compute imports from core

---

## PR-54 — Consolidate Schema/Contract Validation Surfaces

### Goal

Establish a single canonical location for schema validation, eliminating scattered "side registries."

### Current State

`SCHEMA_REGISTRY` lives in `codeintel.config.datasets.schema_registry` and is used by:
- `build/context.py`
- `build/hamilton/contracts/pandera_hook.py`
- `build/plugins/analytics/functions/metrics.py`
- `config/datasets/*.py` (introspection, validation, lineage, etc.)
- `storage/metadata/bootstrap.py`
- `storage/gateway/factory.py`
- `serving/contracts/operation_contract_reflection.py`
- `cli/handlers/ops.py`, `cli/handlers/storage.py`

### Decision Required

**Option A: Keep in `config.datasets` (Current)**
- Pros: Centralized configuration, used by many subsystems
- Cons: Not Hamilton-specific, may drift from build contracts

**Option B: Move to `build.hamilton.contracts`**
- Pros: Aligns with Hamilton-first architecture
- Cons: Requires significant import updates, may over-couple non-build code

**Recommendation:** **Option A** — Keep `SCHEMA_REGISTRY` in `config.datasets` but ensure:
1. Build-layer validation exclusively uses this registry
2. No parallel registries exist
3. Import pattern is consistent

### Tasks

- [ ] Audit for any parallel schema registries
- [ ] Ensure `build/hamilton/contracts/` exclusively uses `SCHEMA_REGISTRY`
- [ ] Add architecture test forbidding alternative registries
- [ ] Document the canonical validation flow

### Tests Checklist

- [ ] `tests/build/hamilton/test_pr54_single_schema_registry.py`
  - [ ] Scan test ensuring no other `*Registry` classes for schemas
  - [ ] Verify `SCHEMA_REGISTRY` is the only schema source

---

## PR-55 — Final Sweep: API Cleanup & Taxonomy

### Goal

Polish the consolidated codebase with consistent public APIs and test taxonomy.

### Tasks

#### Snapshot Manifest Cleanup

- [ ] Review `tests/build/hamilton/snapshots/manifest.yaml`
- [ ] Remove stale mode tags (`phase0` if no longer used)
- [ ] Ensure tags reflect current architecture:
  - `generated` — Hamilton-generated mode
  - `native` — Native implementation
  - `phase4` — Asset catalog features
- [ ] Add missing PR tags for new features

#### Public API Audit

- [ ] Review `src/codeintel/build/hamilton/__init__.py`
  - [ ] Update `__all__` exports
  - [ ] Remove references to deleted modules/modes
- [ ] Review `src/codeintel/build/__init__.py`
  - [ ] Ensure clean public API
  - [ ] Remove deprecated exports

#### Dead Import Sweep

- [ ] Run repo-wide import analysis
- [ ] Fix any imports of removed paths
- [ ] Ensure no circular imports introduced

### Tests Checklist

- [ ] `tests/build/hamilton/test_pr55_snapshot_manifest_taxonomy.py`
  - [ ] Validate all `cases[*].tags` conform to defined taxonomy
  - [ ] Verify no legacy/orphan tags remain
- [ ] Import smoke tests for all public packages

---

## Implementation Timeline

```
Week 1: PR-51 Batch 1 (Low-complexity write migrations)
        ├── 5 modules migrated
        └── Allowlist reduced by 5

Week 2: PR-51 Batch 2 (Medium-complexity migrations)
        ├── 5 modules migrated
        └── Allowlist reduced by 10

Week 3: PR-51 Batch 3 + PR-52
        ├── cfg_dfg migration (6 writes)
        ├── Legacy orchestrator identification
        └── Allowlist at 0

Week 4: PR-53 (Compute consolidation)
        ├── Centrality consolidation
        └── Structural algorithms moved

Week 5: PR-54 + PR-55 (Cleanup)
        ├── Schema registry decision implemented
        ├── Taxonomy cleanup
        └── Final sweep
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Direct writes outside build | 22 | 0 |
| Allowlisted files | 11 | 0 |
| Duplicate compute modules | 2+ | 0 |
| Schema registries | 1 | 1 |
| Legacy tags in manifest | Present | Removed |

---

## Risk Mitigation

### Risk: Breaking Changes During Migration

**Mitigation:**
- Each batch includes integration tests
- Schema validation ensures output compatibility
- Parallel run capability during transition

### Risk: Performance Regression

**Mitigation:**
- Benchmark critical paths before/after
- Hamilton materializers support batching
- Lazy evaluation preserved

### Risk: Incomplete Migration

**Mitigation:**
- Architecture guardrail tests fail on regression
- Allowlist shrinks monotonically
- Weekly progress reviews

---

## Appendix: File-by-File Migration Checklist

### PR-51 Migration Tracking

| File | Status | PR | Notes |
|------|--------|-----|-------|
| `analytics/compute/coverage/functions.py` | ⬜ Pending | | |
| `analytics/compute/data_models/usage.py` | ⬜ Pending | | |
| `analytics/profiles/writer_guard.py` | ⬜ Pending | | |
| `analytics/functions/function_history.py` | ⬜ Pending | | |
| `analytics/history/history_timeseries.py` | ⬜ Pending | | |
| `analytics/dependencies/core.py` | ⬜ Pending | | |
| `analytics/entrypoints/core.py` | ⬜ Pending | | |
| `analytics/data_models/core.py` | ⬜ Pending | | |
| `analytics/parsing/validation.py` | ⬜ Pending | | |
| `analytics/testing/graph_metrics.py` | ⬜ Pending | | |
| `analytics/cfg_dfg/materialize.py` | ⬜ Pending | | |

### Legend
- ⬜ Pending
- 🔄 In Progress
- ✅ Complete
- ❌ Blocked
