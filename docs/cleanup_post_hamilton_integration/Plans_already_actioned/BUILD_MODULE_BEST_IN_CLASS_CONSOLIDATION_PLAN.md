# Build Module Best-in-Class Consolidation Plan

This plan covers the follow-up consolidation opportunities in `src/codeintel/build` identified after
the Hamilton-first migration work. The goal is to **reduce duplication**, **eliminate drift between
APIs that represent the same concept**, and **harden execution correctness** by converging on a small
set of canonical abstractions.

## Goals

- Make the Hamilton-first path the **single, coherent execution system** (one set of invariants,
  one set of factories, one set of persistence rules).
- Eliminate redundant layers that have overlapping responsibilities (runner vs manifest hook,
  registry vs catalog vs graph wrappers, multiple pipeline templates).
- Improve hardness:
  - prevent contract↔DAG drift
  - prevent silent “success” with incomplete/incorrect output metadata
  - avoid SQL injection vectors in internal utilities
  - ensure run lifecycle always completes (start → target records → telemetry → completion)
- Decommission deprecated/legacy code as soon as replacement is working.

## Non-goals

- No new features outside the consolidation scope.
- No “polish-only refactors” that do not reduce duplication or risk.
- No changes to coverage artifacts policy.

## Acceptance gates (must pass for every merged slice)

Run locally after each meaningful slice:

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

## Current pain points (summary)

1. **Duplicate concepts split across modules**
   - Skip logic + manifest persistence + record construction exist in multiple places:
     - `codeintel.build.hamilton.hooks.manifest_hook`
     - `codeintel.build.hamilton.native.runner`
     - `codeintel.build.hamilton.native.executor`
     - `codeintel.build.hamilton.native.materialization_records`
2. **Executor wiring duplicates “hook composition” and contract graph construction**
   - `HamiltonBuildExecutor` composes hooks manually and rebuilds a contract graph.
3. **Multiple registry/catalog APIs compete**
   - `registry.py`, `target_registry.py`, and `target_catalog.py` all provide related “lookup and
     graph” functionality.
4. **Multiple pipeline templates**
   - `executor_pipeline`, `rows_pipeline`, `ibis_pipeline` share a lot of structural glue.
5. **Ad hoc SQL fragments**
   - Even for internal code, string-built queries are showing up and create avoidable risk.
6. **Run persistence spread across multiple components**
   - Run start/complete, per-target persistence, telemetry persistence, and asset emission are
     handled in separate call sites without a single “run writer”.
7. **Legacy OutputTarget construction paths still exist**
   - Multiple ways to construct/validate target specs increases the chance of drift and
     inconsistent invariants.

---

# Workstreams

## Workstream 1: Unify skip + manifest + record creation APIs

### Goal

Create one canonical API surface for:

- computing input hash/options hash
- determining whether a target can be skipped
- creating `TargetRunRecord` for succeeded/skipped/failed
- persisting manifests

### Proposed end-state

Introduce a single module (or tight module cluster) that owns run-record semantics, e.g.:

- `codeintel.build.hamilton.run_records` (preferred)

It should subsume the responsibilities that are currently split across:

- `codeintel.build.hamilton.hooks.manifest_hook`
- `codeintel.build.hamilton.native.runner`

### Implementation steps

1. Define canonical types and their ownership
   - Decide which module is the owner for:
     - `NativeRunInfo`
     - `RunRecordInputs`
     - `SkipCheckRequest`
     - `ManifestSaveRequest`
     - `create_run_record(...)`
     - `save_manifest(...)`
   - Clarify invariants:
     - In strict mode, success records must provide row_counts whose keys exactly match
       `target.contract.table_keys` when the target declares tables.
2. Move/merge functions into the canonical module
   - Keep backwards-compatible re-exports temporarily (thin wrappers).
3. Replace call sites
   - `NativeTargetExecutor` uses canonical skip/record/manifest functions.
   - `materialization_records` uses canonical record construction consistently.
   - Templates (`rows_pipeline`, `ibis_pipeline`, `executor_pipeline`) use canonical record helpers.
4. Delete deprecated modules once fully migrated
   - Remove duplicate helper functions and legacy re-exports.

### Decommission checklist

- Remove the old entry points once no longer imported:
  - `codeintel.build.hamilton.native.runner.should_skip_native_target` (if redundant)
  - `codeintel.build.hamilton.hooks.manifest_hook.should_skip` (if redundant)

---

## Workstream 2: Consolidate executor hook composition and contract graph usage

### Goal

Make hook composition and strict-contract activation consistent across the build system by making
`HamiltonBuildExecutor` delegate to a single “hook factory”.

### Proposed end-state

- `HamiltonBuildExecutor` uses `codeintel.build.hamilton.hooks.build_hooks(...)` to construct
  lifecycle adapters, instead of constructing hook instances itself.
- Eliminate `_build_contract_graph()` and any duplicate “rebuild a base TargetGraph” logic in
  `HamiltonBuildExecutor`.

### Implementation steps

1. Introduce a canonical `HookOptions` mapping from `BuildEnv` / executor config
   - Determine how `env.strict_contracts` maps to `HookOptions(strict_contracts=True)`.
2. Replace `_build_adapters(...)` with `build_hooks(...)`
   - Ensure `NodeTelemetryHook` is created exactly once and flushed reliably.
3. Ensure telemetry flush is exception-safe
   - Use `try/finally` around DAG execution to flush telemetry even when execution fails.
4. Remove dead duplication
   - Delete `_build_contract_graph()` if no longer needed.

### Decommission checklist

- Remove any unused manual hook wiring code in `src/codeintel/build/hamilton/executor.py`.

---

## Workstream 3: Collapse registry/catalog/graph APIs into a single TargetSystem

### Goal

Reduce ambiguity about “where to get target metadata” and “where dependencies come from” by
providing one canonical object that bundles:

- canonical target specs (metadata + contract)
- derived dependency graph (Hamilton-derived)
- indexes (by name, by table_key, by artifact_name)

### Proposed end-state

Introduce a `TargetSystem` concept (name can vary) that becomes the standard entry point:

- `codeintel.build.target_system.load_target_system() -> TargetSystem`
  - `catalog: TargetCatalog`
  - `graph: TargetGraph` (Hamilton-derived)
  - `runtime: HamiltonRuntime` (optional, but useful for introspection + mapping)

Then deprecate overlapping public functions:

- `codeintel.build.registry.get_target_graph`
- ad hoc use of `TargetRegistry` for simple lookups

### Implementation steps

1. Define the `TargetSystem` type
   - It should expose:
     - `get_target(name)`
     - `closure(targets)`
     - `target_for_table_key(table_key)`
     - `target_for_artifact(artifact_name)`
2. Move call sites to `TargetSystem`
   - `HamiltonBuildExecutor` should use `TargetSystem.graph` and `TargetSystem.runtime`.
   - Any schema derivation should source targets from `TargetSystem.catalog.targets`.
3. Preserve compatibility
   - Keep old functions as thin wrappers around `TargetSystem` while migrating call sites.
4. Remove redundant modules/APIs
   - Once all call sites migrate, delete the deprecated wrappers.

---

## Workstream 4: Standardize pipeline templates into a single “materialize template”

### Goal

Reduce repeated glue across:

- `executor_pipeline` (ComputeResult → executor → TargetRunRecord)
- `rows_pipeline` (rows → DuckDBRowsSaver → record)
- `ibis_pipeline` (Ibis expr → DuckDBIbisTableSaver → record)

### Proposed end-state

Create one parameterized template that supports three “compute payload forms”:

1. `dict[str, int]` row-count computation (executor pattern)
2. row tuples (DuckDBRowsSaver)
3. Ibis table expressions (DuckDBIbisTableSaver)

Options:

- If single-table: route through `record_from_duckdb_materialization`
- If multi-table: use a multi-materialization aggregator pattern consistently

### Implementation steps

1. Define a single template module with reusable nodes
   - Prefer names like:
     - `materialize_rows(...)`
     - `materialize_ibis_expr(...)`
     - `materialize_executor_result(...)`
2. Migrate native targets incrementally
   - Start with one module (e.g., analytics) and convert patterns to the unified template.
3. Remove the now-redundant templates
   - Delete `rows_pipeline.py` / `ibis_pipeline.py` / `executor_pipeline.py` once unused.

---

## Workstream 5: Centralize safe SQL query utilities for build-time introspection

### Goal

Stop writing ad hoc SQL snippets in target implementations. Provide hardened helpers for:

- count rows for snapshot
- existence checks (optional)
- delete-by-snapshot (if needed)

### Proposed end-state

Introduce a module such as:

- `codeintel.build.storage_queries` or `codeintel.build.hamilton.sql`

Key properties:

- No dynamic table interpolation with f-strings.
- Use a **whitelist** mapping from known table_key → query string when a table identifier must be
  embedded in SQL.

### Implementation steps

1. Define canonical helpers
   - `count_rows_for_snapshot(gateway, table_key, repo, commit) -> int`
   - `count_rows_for_snapshot(gateway, table_key, snapshot) -> int`
2. Replace ad hoc count queries in native targets
   - Start with the targets that currently execute explicit count queries.
3. Add guardrails
   - Prefer hard failures for unknown table keys in these helpers to prevent accidental drift.

---

## Workstream 6: Consolidate build-run persistence into a BuildRunWriter service

### Goal

Centralize run lifecycle persistence (start → per-node telemetry → per-target records → completion),
ensuring it is:

- consistent
- exception-safe
- easy to extend

### Proposed end-state

Create a `BuildRunWriter` used by:

- `HamiltonBuildExecutor`
- `NodeTelemetryHook` (either directly or through a narrow interface)

It should own:

- starting the run record
- saving per-target records
- saving node telemetry (buffer flush)
- emitting asset catalog records
- completing the run record

### Implementation steps

1. Define `BuildRunWriter` interface and concrete implementation
2. Refactor executor to use the writer
   - Replace `_start_build_run`, `_persist_run_targets`, `_persist_asset_catalog`, `_complete_build_run`
     with methods on the writer.
3. Refactor telemetry hook persistence to call the writer
   - Keep buffering inside the hook, but persistence through the writer.
4. Ensure failure paths still call `complete_run(...)`
   - Especially when closure computation fails or DAG execution throws.

---

## Workstream 7: Remove legacy OutputTarget construction paths and converge on `make_output_target`

### Goal

Ensure there is one canonical way to declare build target metadata next to Hamilton-native
implementations, and reduce alternative constructors that can diverge in behavior.

### Proposed end-state

- New/updated targets must use:
  - `codeintel.build.hamilton.native.target_spec_helpers.make_output_target`
- Legacy constructors in `codeintel.build.targets.OutputTarget` are either:
  - marked internal-only and removed from call sites, or
  - removed if no longer needed.

### Implementation steps

1. Audit all constructions of `OutputTarget(...)` and `OutputTarget.from_tables(...)`
2. Migrate call sites to `make_output_target(...)` or the catalog loader
3. Remove or quarantine legacy factories
   - If they remain, ensure they route through the same validation logic as the canonical path.

---

# Execution plan (order of operations)

This order minimizes risk and keeps the system shippable at each step:

1. **Workstream 1** (unify run-record/manifest/skip) — foundational, reduces drift everywhere.
2. **Workstream 2** (executor uses hook factory) — reduces duplication and hardens run lifecycle.
3. **Workstream 6** (BuildRunWriter) — centralizes persistence, reduces scattered failure handling.
4. **Workstream 5** (safe SQL utilities) — removes injection risk and normalizes DB introspection.
5. **Workstream 3** (TargetSystem) — simplifies APIs and eliminates overlapping registries.
6. **Workstream 4** (template unification) — reduces code volume and standardizes patterns.
7. **Workstream 7** (legacy target factories removal) — final cleanup once new path is dominant.

Each workstream should land as a sequence of small slices with the acceptance gates passing after
each slice.

# Deliverables checklist

- [ ] Canonical run-record + manifest API (single module)
- [ ] `HamiltonBuildExecutor` delegates hook wiring to `build_hooks`
- [ ] Safe SQL introspection helpers; no remaining f-string table interpolation
- [ ] BuildRunWriter service for run lifecycle persistence
- [ ] TargetSystem replaces registry/catalog overlap
- [ ] Unified materialize template replaces duplicated pipeline templates
- [ ] Legacy constructors removed or fully quarantined

