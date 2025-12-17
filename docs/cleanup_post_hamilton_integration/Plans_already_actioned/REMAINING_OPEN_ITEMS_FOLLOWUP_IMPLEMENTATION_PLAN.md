<!--
This document is a follow-up to REMAINING_OPEN_ITEMS_PLAN.md.
It captures the *remaining literal items* plus a small set of optional hardening tasks.
-->

# Remaining Open Items — Follow-Up Implementation Plan

> **Status**: Proposed implementation plan (ready to execute)  
> **Author**: AI Assistant  
> **Date**: 2025-12-17  
> **Audience**: Build/infra engineers and agents working in `src/codeintel/build/`  
> **Scope**: Finalize the remaining items after executing `REMAINING_OPEN_ITEMS_PLAN.md`

---

## Executive Summary

`docs/cleanup_post_hamilton_integration/REMAINING_OPEN_ITEMS_PLAN.md` is now effectively executed:

- Phase 7 (Registry unification) is implemented via `TargetRegistry` + `runtime.graph` as canonical.
- Phase 5 success criteria is met by adding DAG-visible `@pipe_input` steps to two additional targets.
- Phase 9 (parallel promotion) is implemented via CLI docs + tests; benchmarks remain optional.
- Phase 6 remains optional and is left as-is.

What remains is primarily **cleanup/hardening** to remove legacy ambiguity and prevent regression:

1. **(Mandatory)** Fully decommission static `dependencies=` declarations in
   `src/codeintel/build/registry.py` so there is no longer any “dual source of truth”.
2. **(Optional)** Convert the originally suggested `external_deps` / `function_contracts` targets to
   `@pipe_input` if the additional DAG visibility is still desired.
3. **(Optional)** Add a minimal benchmark harness for parallel execution to quantify speedups.

---

## Current State (Post-Implementation Snapshot)

Canonical dependency/topology source:
- `codeintel.build.hamilton.driver_factory.build_driver()` returns a `HamiltonRuntime` where
  `runtime.graph` contains **Hamilton-derived** dependencies.
- `codeintel.build.registry.get_target_graph()` returns `runtime.graph`.

Key consolidation artifacts:
- Unified wrapper: `../../src/codeintel/build/target_registry.py`
- Dependency derivation + graph build: `../../src/codeintel/build/hamilton/introspect.py`
- DAG-visible `@pipe_input` expansions:
  - `../../src/codeintel/build/hamilton/native/analytics/coverage_targets.py`
  - `../../src/codeintel/build/hamilton/native/graphs/graph_targets.py`
- Parallel CLI + tests:
  - `../../src/codeintel/cli/commands/build.py`
  - `../../src/codeintel/cli/handlers/build.py`
  - `../../tests/cli/test_build_parallel.py`

---

## Remaining Items (What’s Still Open)

### 1) Mandatory: Decommission Static Dependencies in `registry.py` (Phase 7.4)

**Goal**: Make Hamilton DAG-derived dependencies the only dependency source; remove stale/duplicated
metadata that can drift and confuse future refactors.

**Why this still matters (even though correctness is already fixed)**:
- `../../src/codeintel/build/registry.py` still encodes dependency tuples on `OutputTarget` constants.
- Even if not used in the primary runtime path today, it invites accidental reuse in future code and
  breaks the “single source of truth” principle.

#### Implementation Plan

1. **Audit dependency usage and enforce boundaries**
   - Search for dependency reads that bypass `TargetGraph`:
     - `rg "\\.dependencies" src/codeintel/build`
     - `rg "ALL_TARGETS" src/codeintel/build`
   - Categorize call sites:
     - **Allowed**: `target.dependencies` where `target` comes from `runtime.graph` / `get_target_graph()`.
     - **Not allowed**: `target.dependencies` where `target` comes from `ALL_TARGETS` (static registry).
   - If any “not allowed” sites exist:
     - Replace with `graph.dependencies_of(target_name)` or `graph.get(target_name).dependencies`
       using the canonical graph.
     - Prefer APIs that accept a `TargetGraph`/`TargetRegistry` rather than raw `OutputTarget`.

2. **Strip static `dependencies=` in `../../src/codeintel/build/registry.py`**
   - Set every `OutputTarget(..., dependencies=...)` to `dependencies=()`.
   - Keep the `OutputTarget` constants as the canonical metadata source (contract/resources/description),
     but treat dependency topology as DAG-derived only.

3. **Add a regression test to prevent reintroduction**
   - Add `tests/build/test_registry_has_no_static_dependencies.py`:
     - Assert `all(target.dependencies == () for target in ALL_TARGETS)`.
   - Keep the existing Hamilton-vs-graph dependency parity test(s) as the correctness gate.

4. **Add a runtime invariant (optional but recommended)**
   - In `build_driver()`:
     - Validate that all `ALL_TARGETS` have `dependencies == ()`.
     - Fail fast with a clear error if violated (prevents silent drift).

#### Acceptance Criteria

- `ALL_TARGETS` contains **no** non-empty dependency tuples.
- All closure planning, state computation, and hashing use **Hamilton-derived** dependencies via
  `runtime.graph`.
- Quality gates:
  - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
  - `uv run pytest -q`

#### Risks & Mitigations

- **Risk**: A code path still uses `ALL_TARGETS` (static) targets for hashing or closure.
  - **Mitigation**: The audit step + regression test + fast-fail invariant.

---

### 2) Optional: Finish the “Suggested” `@pipe_input` Conversions (Phase 5 candidates)

Phase 5’s success criteria is already met, but the original doc suggested two specific candidates:

- `external_deps` (analytics dependency analysis)
- `function_contracts` (function pre/postcondition inference)

If these targets are still expected to be “first-class DAG citizens” with inspectable intermediate
steps, implement the following.

#### Implementation Plan

1. **Decide if DAG-level step visibility is actually valuable for each target**
   - If the target is mostly Python/Ast logic (not Ibis/pandas pipeline), `@pipe_input` may not be a
     net win; prefer explicit helper functions + unit tests instead.

2. **If converting: move step functions to shared compute modules**
   - Follow the established consolidation approach used for coverage:
     - Shared steps live in `codeintel/analytics/compute/...` (pure, typed, reusable).
     - Hamilton target modules become thin wiring layers.

3. **Implement `@pipe_input` wiring**
   - Ensure each step is:
     - Pure (no storage IO).
     - Typed and independently testable.
     - Named/stable so DAG diffs remain readable.

4. **Add tests**
   - Snapshot or structural test that the step nodes appear in the Hamilton graph.
   - Golden-data test (where feasible) to ensure output stability.

#### Acceptance Criteria

- Step nodes for the target appear in the DAG export and are tagged consistently.
- `uv run pytest -q` passes with no output changes (unless intentionally revised).

---

### 3) Optional: Parallel Execution Benchmark Harness (Phase 9.3)

Parallel execution is already supported and tested. If you want quantitative confidence and
regression detection:

#### Implementation Plan

1. **Pick the benchmark shape**
   - Prefer a small deterministic benchmark suite that:
     - Runs on CI reliably (short, stable).
     - Separates “I/O bound” vs “CPU bound” workloads.

2. **Implement as opt-in**
   - Use a marker (e.g., `@pytest.mark.benchmark`) and exclude by default.
   - Provide a small script or Makefile target to run locally.

3. **Metrics**
   - Compare sequential vs `threadpool` on a fixed snapshot/fixture.
   - Track speedup ratio and absolute wall time.

#### Acceptance Criteria

- Benchmark suite runs locally without flakiness.
- Documentation explains how to run and interpret results.

---

## Recommended Execution Order

1. **Mandatory**: Decommission static dependencies in `registry.py` + add regression test(s).
2. Optional: `@pipe_input` conversions for `external_deps` / `function_contracts` (only if valuable).
3. Optional: Parallel benchmark harness.

---

## Quality Gates (Non-Negotiable)

Run after each logical chunk (at minimum before merging):

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

