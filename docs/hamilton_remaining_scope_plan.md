# Hamilton Remaining Scope Plan (Parts A-C)

## Overview

This plan captures the remaining implementation scope from:
- `docs/Make_Hamilton_graph_authoritative.md` (Part A)
- `docs/Make_Hamilton_graph_authoritative_partB.md` (Part B)
- `docs/Make_Hamilton_graph_authoritative_partC.md` (Part C)

The core code for DagCatalog and parameterized support nodes is already present. The remaining
scope is primarily (1) documentation parity, (2) cache-first incrementality with manifests
as audit-only artifacts, and (3) static enforcement of the single composition root rule.

## Scope

In scope:
- Documentation parity for the "Hamilton graph is authoritative" architecture.
- Cache-first incrementality with Hamilton cache as the sole skip/compute authority.
- Static linting to prevent runtime/driver construction inside DAG node code.

Out of scope:
- Part D relation-first refactor and Ibis removal.
- Meta schema plan (separate document and phases).

## Phase 0: Baseline Audit (fast, deterministic)

Objective: validate the current repo state against Parts A-C before changing behavior.

Steps:
1. Scan for remaining TargetGraph references in non-archive docs and note required updates.
2. Identify all manifest-driven control flow and state computation callsites.
3. Confirm that planning paths use cache probes (not manifests) and that compose_runtime is
   already guarded at runtime.

Deliverables:
- A short audit note attached to this plan (append a "Status" section if needed).

Acceptance:
- Every remaining control-flow use of manifests is enumerated with file paths.

## Phase 1: Part A Documentation Parity (no code changes)

Objective: align architecture and plan docs with the current DagCatalog-based implementation.

Tasks:
1. Update core architecture docs that still describe TargetGraph or TagIndex:
   - `docs/architecture_ph6.md`
   - `docs/architecture.md`
   - `docs/scip_indexing_upgrade_plan.md`
   - `docs/full_dag_basis_implementation_plan.md`
2. Replace TargetGraph references with DagCatalog descriptions, including:
   - Target anchor discovery and closure computation.
   - Support node expansion using parameterization in `support_nodes.py`.
   - Tag discovery via `Driver.list_available_variables(tag_filter=...)` or TagQuery helpers.
3. If any archived docs are updated, add a "Historical note" block clarifying that the content
   reflects pre-DagCatalog architecture.

Deliverables:
- Docs updated to reflect the single-source DagCatalog model.

Acceptance:
- No TargetGraph or TagIndex references in non-archive docs.
- All examples reference DagCatalog, TagQuery, or tag_filter usage.

## Phase 2: Part B Cache-First Incrementality (manifest audit only)

Objective: make Hamilton caching the sole authority for skip/compute decisions while keeping
manifests as audit records.

### 2.1 Define the Cache Authority Contract

Tasks:
1. Document the cache authority contract in this plan (or a short design note):
   - Cache key is the authoritative "input_hash" for targets.
   - Data version is the authoritative "output identity".
2. Align TargetRunRecord fields with cache terminology:
   - `input_hash` should map to the resolved cache key for the target.
   - Optionally attach data version to run metadata or decision trace.

Files to touch:
- `src/codeintel/build/hamilton/run_records.py`
- `src/codeintel/build/hamilton/decision_trace.py` (if data version is surfaced)

### 2.2 Replace Manifest-Driven State and Skip Logic

Tasks:
1. Rework state computation to use CacheIndex/CacheStore:
   - `src/codeintel/build/state.py`
   - `src/codeintel/build/state_computer.py`
   - `src/codeintel/build/session.py`
2. Remove manifest-based "missing" and "stale" determinations from control flow.
3. Keep manifest reads only in audit/reporting modules, not in execution gating.

Deliverables:
- State computations use cache presence and data version, not manifest presence.

### 2.3 Executor Alignment

Tasks:
1. Remove manifest-based input hash usage in executor:
   - `_safe_input_hash` should use cache keys (via CacheKeyResolver/CacheStore).
   - `manifest_index` should not be required for execution decisions.
2. Ensure `_apply_cache_keys` is the single point of record input_hash update.
3. Keep CacheManifestWriter emission as audit-only; no control flow reads.

Files to touch:
- `src/codeintel/build/hamilton/executor.py`
- `src/codeintel/build/hamilton/cache_key_resolver.py`
- `src/codeintel/build/run_context.py`
- `src/codeintel/build/hamilton/env.py`
- `src/codeintel/cli/handlers/build.py` (remove manifest_index wiring for control flow)

### 2.4 Planning and Explain Outputs

Tasks:
1. Validate plan/explain nodes rely only on CacheIndex and cache probes.
2. Remove any manifest-based hit/miss logic in planning outputs (if present).

Files to touch (if needed):
- `src/codeintel/build/hamilton/native/planning/plan_nodes.py`
- `src/codeintel/build/hamilton/planner.py`

### 2.5 Manifest Layer Becomes Audit-Only

Tasks:
1. Explicitly mark manifest writer/reader as audit output, not control-plane input.
2. Ensure any manifest reads are used only for reporting/decision trace export.

Files to touch:
- `src/codeintel/build/hamilton/cache_adapter.py`
- `src/codeintel/build/hamilton/decision_trace.py`
- `src/codeintel/build/hamilton/native/export/decision_trace.py`

### 2.6 Test and Validation Updates

Tasks:
1. Update existing tests that assume manifest-based state:
   - `tests/build/test_state.py`
   - `tests/build/test_state_computer.py`
2. Add tests that assert cache-based state and skip logic.
3. Add coverage for cache-key-driven TargetRunRecord updates.

Acceptance:
- No execution or planning path depends on `manifest_index` for control flow.
- CacheIndex/CacheStore presence is sufficient to compute target state.
- Manifests are emitted and consumed only by audit/reporting code.

## Phase 3: Part C Static Enforcement (Single Composition Root)

Objective: prevent runtime/driver construction inside DAG nodes via static linting.

### 3.1 Implement the Linter

Tasks:
1. Create `tools/lint_no_driver_build_in_nodes.py` that scans DAG node directories:
   - `src/codeintel/build/hamilton/native`
   - Any other node packages used for execution
2. AST-check for disallowed imports or calls:
   - `compose_runtime`
   - `build_runtime`
   - `build_driver`
   - `build_runtime_primitives`
   - `RuntimeBundle` construction (direct instantiation inside nodes)
3. Allowlist non-node modules if needed (explicit allowlist in linter).

### 3.2 Wire into Tests and CI

Tasks:
1. Add pytest gate: `tests/lint/test_no_driver_build_in_nodes.py`.
2. Ensure the linter runs in normal test flows (no separate runner required).

### 3.3 Documentation Touchups

Tasks:
1. Update `docs/Make_Hamilton_graph_authoritative_partC.md` with the linter location
   and enforcement guarantee.

Acceptance:
- Linter passes in CI and fails on forbidden driver construction inside DAG nodes.
- Runtime guard remains in `src/codeintel/runtime/compose.py` as a second line of defense.

## Sequencing and Milestones

Recommended order:
1. Phase 0 audit
2. Phase 1 documentation parity
3. Phase 2 cache-first incrementality
4. Phase 3 static enforcement

## Quality Gates

Required checks for each implementation phase:
```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q tests/lint
uv run pytest -q tests/build
```

## Definition of Done

- Docs match the DagCatalog-first architecture and no longer reference TargetGraph or TagIndex.
- Skip/compute decisions are cache-first; manifests are audit-only.
- Static linter prevents runtime/driver construction inside DAG node code.
