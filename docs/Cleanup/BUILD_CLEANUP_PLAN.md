# Build System Cleanup Plan (`src/codeintel/build/**`)

This document is an implementation plan for cleaning up deprecated/compatibility/dead code in the
build system after recent refactors.

## Goals

- Remove clearly unused modules and stubs.
- Eliminate mismatches between the build target registry and plugin registry.
- Reduce compatibility shims where they no longer provide value.
- Make `OutputContract` the single source of truth for “what a target produces”, and remove the
  remaining “legacy table key” fallbacks.
- Keep the codebase passing all quality gates (Ruff, Pyright, Pyrefly, pytest).

## Non-goals

- No functional changes to build outputs beyond removing legacy behaviors and enforcing existing
  contracts more consistently.
- No re-architecture of the build system; this is “delete + simplify + align” work.

## Preconditions / setup

1. Ensure a clean working tree (recommended):
   - `git status --porcelain`
2. Bootstrap and sync deps:
   - `scripts/bootstrap.sh`
   - `uv sync`
3. Capture a baseline quality report + tests:
   - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
   - `uv run pytest -q`

## Work items (recommended execution order)

### 1) Delete “high-confidence unused” code

These are items with no in-repo call sites and no observable behavior impact.

#### 1.1 Remove `RunEnvironment` if it’s truly unused

- **Candidate**: `src/codeintel/build/environment.py`
- **Rationale**: `RunEnvironment` is only referenced within the file (no imports elsewhere).
- **Implementation**
  1. Confirm no usage (including docs/scripts):
     - `rg -n "RunEnvironment|codeintel\\.build\\.environment" -S src tests docs scripts`
  2. Decide the intended direction:
     - **Option A (recommended): delete** the module entirely.
     - **Option B: wire it into run tracking** (e.g., record `tool_versions` and config hash when
       `BuildRunRecord` is started/completed). If choosing this path, define the persistence schema
       and update the storage layer accordingly; this is strictly more work than deletion.
  3. If deleting:
     - Remove `src/codeintel/build/environment.py`.
     - Remove any exports/references if added later.
  4. Run quality gates (see “Validation” at end).
- **Acceptance criteria**
  - No import paths mention `codeintel.build.environment`.
  - Quality report passes; pytest passes.

#### 1.2 Remove unused compatibility re-export `codeintel.build.datasets`

- **Candidate**: `src/codeintel/build/datasets.py`
- **Rationale**: Zero in-repo imports; type now lives under `codeintel.build.hamilton.io`.
- **Implementation**
  1. Confirm no usage:
     - `rg -n "codeintel\\.build\\.datasets" -S src tests docs`
  2. If empty:
     - Delete `src/codeintel/build/datasets.py`.
  3. Run quality gates.
- **Acceptance criteria**
  - No references remain.
  - Quality report passes; pytest passes.

### 2) Fix plugin registry / target registry mismatches

#### 2.1 Remove dead plugin target `graph_metrics_secondary`

- **Candidates**
  - Registry entry: `src/codeintel/build/plugin_registry.py` (`graph_metrics_secondary`)
  - Implementation: `src/codeintel/build/plugins/graphs/metrics/secondary.py`
  - Exports: `src/codeintel/build/plugins/graphs/metrics/__init__.py`,
    `src/codeintel/build/plugins/graphs/__init__.py`
- **Rationale**
  - `graph_metrics_secondary` is present in plugin definitions but there is no corresponding
    `OutputTarget` in `src/codeintel/build/registry.py`.
  - The plugin implementation is currently a no-op stub that returns zeros.
- **Implementation**
  1. Confirm the mismatch (optional scripted check):
     - Compare `plugin_registry._PLUGIN_DEFINITIONS` targets against `get_target_graph()` targets.
  2. Remove the plugin target:
     - Delete the `("graph_metrics_secondary",)` entry from `_PLUGIN_DEFINITIONS`.
     - Delete `src/codeintel/build/plugins/graphs/metrics/secondary.py`.
     - Remove `SecondaryMetricsPlugin` exports from:
       - `src/codeintel/build/plugins/graphs/metrics/__init__.py`
       - `src/codeintel/build/plugins/graphs/__init__.py`
  3. Update any docs that mention it (optional, since docs may be archival):
     - `rg -n "graph_metrics_secondary|SecondaryMetricsPlugin" -S docs`
  4. Add a regression test to prevent reintroducing mismatches:
     - New test should assert: every plugin-registry target exists in the target graph.
       (It should allow plugin-less native targets, since those are *targets* but not *plugins*.)
  5. Run quality gates.
- **Acceptance criteria**
  - No code references to `graph_metrics_secondary` or `SecondaryMetricsPlugin`.
  - New test enforces registry consistency.
  - Quality report passes; pytest passes.

### 3) Collapse “compatibility re-export” modules (decision-based)

These files exist only to re-export canonical types from new locations:

- `src/codeintel/build/env.py` → `codeintel.build.hamilton.env.BuildEnv`
- `src/codeintel/build/snapshot.py` → `codeintel.config.primitives.SnapshotRef`
- `src/codeintel/build/artifacts.py` → `codeintel.build.hamilton.io.artifact_ref.ArtifactRef`
- `src/codeintel/build/hamilton/dataset_ref.py` → `codeintel.build.hamilton.io.dataset_ref.*`
- `src/codeintel/build/hamilton/artifact_ref.py` → `codeintel.build.hamilton.io.artifact_ref.*`

**Decision point**: are these import paths part of your supported public API?

- If **yes**, keep them for now but plan a deprecation window:
  - Keep modules, but add a clear deprecation policy (docs/release notes). Consider a runtime
    warning only if your project standards allow it; otherwise, document deprecation without
    warnings.
- If **no**, remove them after updating internal imports.

#### 3.1 Update internal imports to canonical locations

- **Implementation**
  1. For each shim, search current usage:
     - `rg -n "codeintel\\.build\\.(env|snapshot|artifacts|datasets)" -S src tests`
     - `rg -n "codeintel\\.build\\.hamilton\\.(dataset_ref|artifact_ref)" -S src tests`
  2. Replace imports to canonical modules:
     - Prefer `from codeintel.build.hamilton.env import BuildEnv`
     - Prefer `from codeintel.config.primitives import SnapshotRef`
     - Prefer `from codeintel.build.hamilton.io.artifact_ref import ArtifactRef`
     - Prefer `from codeintel.build.hamilton.io.dataset_ref import DatasetRef, refs_from_target_result, refs_to_tuple`
  3. Ensure types remain available where needed (update `__all__` in public packages if desired).
  4. Run quality gates.

#### 3.2 Delete the shim modules (if not public API)

- **Implementation**
  1. Delete shim modules listed above (including `src/codeintel/build/datasets.py` if not already
     deleted in 1.2).
  2. Run quality gates.
- **Acceptance criteria**
  - No internal references rely on shim modules.
  - Quality report passes; pytest passes.

### 4) Remove “legacy table key” fallback behavior (larger refactor)

This is the highest-value cleanup, but also the most behaviorally sensitive.

#### 4.1 Define the intended semantics (target outputs)

**End-state recommendation**

- A target’s produced tables are defined by `target.contract.table_keys` only.
- Artifact-only targets produce zero tables.
- Any write to a table not present in the contract is a contract violation when validation is on.

This implies removing or neutralizing the legacy fallback behavior of:

- `OutputTarget.table_keys` manufacturing a default `schema_prefix.target_name` key.
- `TargetExecutionContext.write_table()` bypassing schema lookup when `table_key` is in
  `target.table_keys` but not in the contract.
- Various call sites using `contract.table_keys or target.table_keys`.

#### 4.2 Implementation steps

1. **Update `OutputTarget.table_keys`**
   - `src/codeintel/build/targets.py`
   - Change the property to return only `self.contract.table_keys` (no module-based fallback).
   - Ensure artifact-only targets return `()`.
2. **Update call sites to use contract table keys**
   - Replace these patterns:
     - `target.contract.table_keys or target.table_keys`
     - `target.table_keys` (where it implicitly meant “produced tables”)
   - Primary candidates:
     - `src/codeintel/build/context.py` (validation logic)
     - `src/codeintel/build/operations.py` (table→target index)
     - `src/codeintel/build/registry.py` (`get_all_target_table_keys`, `get_target_by_table`)
     - `src/codeintel/build/hamilton/observability.py`
     - `src/codeintel/build/hamilton/planner.py`
     - `src/codeintel/build/hamilton/nodes/node_factory.py` (dataset node generation)
3. **Tighten `TargetExecutionContext.write_table`**
   - `src/codeintel/build/context.py`
   - Remove the “legacy tables bypass schema lookup” path; if `schema is None` and `validate=True`,
     raise `SchemaNotFoundError` (or a more specific contract error if you prefer).
4. **Update tests**
   - Remove/replace any tests that validate “legacy bypass” behavior.
     - Example: `tests/build/test_context_contracts_errors.py` has a test that asserts bypass works.
   - Ensure existing build target definitions all declare schemas for produced tables (they mostly
     do via `get_table_schemas()`).
5. **Add regression tests**
   - Add/adjust tests to ensure:
     - Artifact-only targets have no produced tables.
     - Every table write performed by a plugin is within its contract (where feasible to test).
6. **Run quality gates**

#### 4.3 Risk management / rollout options

- **Option A (single PR)**: do all changes in one go, update tests, and enforce contract-only table
  semantics immediately.
- **Option B (two-step rollout)**:
  1. Update call sites to prefer `contract.table_keys` everywhere while leaving
     `OutputTarget.table_keys` fallback temporarily (internal compatibility).
  2. In a follow-up PR, remove the fallback + legacy bypass once downstream code is migrated.

### 5) Remove redundant/unreachable compatibility branches

These are small cleanups that reduce noise and simplify reasoning.

#### 5.1 Remove unreachable “no contract defined” branches in contract enforcer

- **Candidate**: `src/codeintel/build/hamilton/contracts/enforcement.py`
- **Rationale**: `OutputTarget.contract` is always an `OutputContract` instance; it is always truthy.
- **Implementation**
  - Remove the `if not cls._current_target.contract:` branches in:
    - `validate_table_write`
    - `validate_artifact_write`
  - Ensure the logic still allows empty contracts (empty outputs) appropriately.
  - Run quality gates.

#### 5.2 Remove redundant `if not target.contract` checks in native outputs helpers

- **Candidate**: `src/codeintel/build/hamilton/native/outputs.py`
- **Rationale**: `target.contract` always exists; the meaningful check is whether it contains
  `table_keys` or `artifacts`.
- **Implementation**
  - Simplify:
    - `if not target.contract or not target.contract.table_keys:` → `if not target.contract.table_keys:`
    - `if not target.contract or not target.contract.artifacts:` → `if not target.contract.artifacts:`
  - Run quality gates.

## Validation (must pass before merge)

Run these locally after each major phase (at minimum after phases 1, 2, and 4):

- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- `uv run pytest -q`

Optional targeted tests for faster iteration during refactors:

- `uv run pytest -q tests/build`
- `uv run pytest -q tests/build/hamilton`

