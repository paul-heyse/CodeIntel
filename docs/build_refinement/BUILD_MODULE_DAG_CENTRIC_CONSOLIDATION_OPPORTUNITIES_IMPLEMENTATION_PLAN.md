# BUILD MODULE — DAG‑CENTRIC CONSOLIDATION OPPORTUNITIES (IMPLEMENTATION PLAN)

## Context
This plan captures a set of consolidation opportunities in `src/codeintel/build` that aim to:

- Reduce the number of “ways to do the same thing”.
- Make the Hamilton DAG the single source of truth for dependencies and execution.
- Harden boundaries between build orchestration and storage I/O.
- Improve extensibility and maintainability while preserving (or increasing) functionality.

This plan is intentionally **DAG-first**: native Hamilton modules + templates are treated as the
primary system boundary, and everything else becomes an adapter around that core.

## Success Criteria (Definition of Done)
Across the scope below:

- There is **one canonical API** for accessing target metadata/runtime/graph (no overlapping
  registries/catalogs in active use).
- All native targets in `src/codeintel/build/hamilton/native/*` follow one of a small set of
  **canonical Hamilton patterns** (single-table, multi-table, tool/artifact, views).
- Tagging and naming conventions are **uniform** and enforced by guardrails.
- Build→storage write behavior goes through **Warehouse / saver boundaries** (no ad-hoc writes).
- Planning/state/observability are unified around a single model and do not drift.
- Legacy/compatibility code introduced during migration is **fully removed** by the end.
- Quality gates pass:
  - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
  - `uv run pytest -q`

## Guiding Principles
- **Hamilton DAG is the source of truth**: dependencies are derived, never duplicated in metadata.
- **Single canonical entrypoints**: one “right import” per concern.
- **Hard boundaries**:
  - Build nodes compute; storage writes happen via Warehouse/savers.
  - Runtime-safe annotations only where Hamilton interprets types.
- **Delete deprecated code early**: once migrated, remove the old path to avoid confusion.
- **Guardrails over conventions**: add fast checks to prevent drift.

---

## Workstream A — Consolidate target metadata entrypoints (one canonical API)

### Goal
Eliminate overlapping APIs so the codebase has exactly one “target system” abstraction:

- Runtime (Hamilton driver + mappings)
- TargetGraph (Hamilton-derived dependencies)
- Indexes (by name / table_key / artifact_name)

### Current state
Multiple entrypoints overlap in responsibilities:
- `src/codeintel/build/target_system.py` (already close to the desired end state)
- `src/codeintel/build/registry.py`
- `src/codeintel/build/target_catalog.py`
- `src/codeintel/build/hamilton/native/registry.py`

### Proposed end state
- `TargetSystem` in `src/codeintel/build/target_system.py` becomes the **only** supported access
  point for target runtime + graph + indexes.
- The other modules are either:
  - Deleted, or
  - Reduced to **tiny** compatibility shims that delegate to `TargetSystem` (with a clear
    deprecation window, and then deletion).

### Implementation steps
1. Inventory imports of the “old” entrypoints:
   - Find all references to `codeintel.build.registry.get_target_graph`
   - Find all references to `codeintel.build.target_catalog.load_target_specs/load_target_catalog`
   - Find all references to `codeintel.build.hamilton.native.registry.*`
2. Decide (and document) the canonical surfaces:
   - `load_target_system() -> TargetSystem`
   - `TargetSystem.graph`, `TargetSystem.runtime`, `TargetSystem.get_target(...)`,
     `TargetSystem.target_for_table_key(...)`, `TargetSystem.target_for_artifact(...)`
3. Migrate call sites to `TargetSystem` and delete/replace duplicates.
4. Add a guardrail to prevent reintroducing non-canonical imports.

### Acceptance gates
- `load_target_system()` is used by all core consumers (CLI, schema provider, validation, planner).
- No remaining imports of the deprecated entrypoints.
- Guardrails enforce allowed import set.

---

## Workstream B — Standardize native target implementations onto canonical Hamilton patterns

### Goal
Reduce per-target bespoke wiring. Every native target should use one of a small number of patterns
implemented as reusable templates.

### Canonical patterns (target-level)
1. **Single-table dataset materialization**
   - Compute: `ir.Table` (or rows) with stable schema
   - Materialize: one saver node
   - Record: `TargetRunRecord` via `record_from_duckdb_materialization`
2. **Multi-table dataset materialization**
   - Compute once, extract N tables/row sets, materialize N savers
   - Record aggregated via `record_from_duckdb_materializations`
3. **Tool/artifact targets**
   - Compute tool output (bytes/str/path)
   - Save via `FileArtifactSaver`
   - Record via `record_from_file_artifact_materialization`
4. **Views/materialized views**
   - Build view expressions; materialize with the same saver interface used for datasets
   - Record as for dataset materializations

### Current state
Templates exist but are not consistently used:
- `src/codeintel/build/hamilton/templates/materialize_template.py`
- `src/codeintel/build/hamilton/templates/multi_table_pipeline.py`
- `src/codeintel/build/hamilton/templates/tool_pipeline.py`

Native modules sometimes implement their own variants of “skip+record+save” glue.

### Proposed end state
- Native targets are “thin”: business logic + minimal wiring.
- Templates own the orchestration glue.
- “One way to do it” per pattern.

### Implementation steps
1. Create/extend a single templates index that explicitly names supported patterns:
   - e.g., `codeintel.build.hamilton.templates` re-exports + short aliases
2. For each native target module:
   - Categorize the target into one canonical pattern.
   - Refactor to use the corresponding template.
   - Delete bespoke glue once migrated.
3. Add guardrails enforcing:
   - No ad-hoc “record building” in native modules when a template exists.
   - No duplication of skip-check logic outside `NativeTargetExecutor`/templates.

### Acceptance gates
- Every target module has clear pattern adherence.
- Reduced boilerplate in `src/codeintel/build/hamilton/native/*`.
- Guardrails prevent drift.

---

## Workstream C — Unify tagging + naming conventions and enforce via guardrails

### Goal
Avoid tag drift and make tag-based discovery reliable (for validation, semantic compilation, UI,
etc.).

### Current state
Mixed usage:
- `hamilton.function_modifiers.tag` is used directly in some modules.
- Wrapper helpers exist in `src/codeintel/build/hamilton/tagging.py`.

### Proposed end state
- All build DAG modules use `codeintel.build.hamilton.tagging` helpers:
  - `tag_compute`, `tag_materialize`, `tag_tool`, `tag_dataset`, `tag_artifact`
- Tag keys and node types always align with `codeintel.core.hamilton.tags`.

### Implementation steps
1. Migrate all native modules to wrapper tagging.
2. Add guardrails:
   - Ban direct imports of `hamilton.function_modifiers.tag` in build DAG modules.
   - Optionally ban use of raw tag string keys outside `codeintel.core.hamilton.tags` and
     `codeintel.build.hamilton.tagging`.
3. Add/extend unit tests that scan `src/codeintel/build/hamilton/native` for forbidden patterns.

### Acceptance gates
- Tag discovery in `build/hamilton/validate.py` and semantic compilation remains stable.
- Guardrails reject drift immediately.

---

## Workstream D — Make the build→storage materialization boundary uniform

### Goal
Ensure “write paths” are consistent, auditable, and safe:

- Build does not call storage policy write APIs directly.
- Build does not reach into raw ibis table access except via well-known seams.
- Materialization metadata shapes are stable at Hamilton boundaries.

### Proposed end state
- Warehouse is the canonical write boundary (or savers that delegate to Warehouse).
- A small allowlist exists for the *only* places where `.ibis.table(...)` is allowed (facade/adapter).
- A single helper exists to build `MaterializeOptions` consistently (snapshot/mode/owner_target).

### Implementation steps
1. Make `MaterializeOptions` creation uniform:
   - Introduce `materialize_options(env, target_name, *, mode=..., snapshot=...)`.
2. Centralize any remaining “write policy decisions” in Warehouse (not build).
3. Guardrails:
   - Ban `.policy.(delete_for_snapshot|bulk_insert*|delete)(` under `src/codeintel/build`.
   - Ban `.ibis.table(` under `src/codeintel/build` except approved seams.
4. Add tests enforcing these invariants (fast regex scan).

### Acceptance gates
- A code reviewer can answer “how does data get written?” by reading Warehouse + savers only.
- Guardrails + tests keep the boundary intact.

---

## Workstream E — Unify options loading (fix plan/execution drift)

### Problem
Planning computes an options hash, but many targets instantiate options via `Options()` and do not
consume config → options at runtime. This risks:
- “Recompute when nothing changes”
- “Don’t recompute when behavior changes”

### Proposed end state
- A single “options loading” utility exists:
  - `load_target_options(env, target_name, OptionsType)` (or similar)
- Options dataclasses can be populated from `env.config.parameters_for(target_name)` deterministically.
- The same options object used for compute is used for `options_hash` computation.

### Implementation steps
1. Introduce a canonical options loading API in build/hamilton:
   - Responsible for extracting per-target config section and validating types.
2. Standardize each target options dataclass to support deterministic construction:
   - `@classmethod from_parameters(params: TargetParameters) -> Options`
3. Update native targets to call `load_target_options(...)` rather than `Options()`.
4. Ensure planner hashes exactly what execution consumes.
5. Guardrails:
   - Ban direct `Options()` instantiation in native modules (allow in tests).

### Acceptance gates
- Changing `codeintel.build.toml` yields predictable recompute behavior.
- Options hashing explains “why will this run?” deterministically.

---

## Workstream F — Unify planning, state, and observability around one model

### Goal
Reduce duplicated concepts (“state”, “plan”, “observability”) to one canonical model that the CLI
and internal tools use consistently.

### Current state
- State computation: `src/codeintel/build/state_computer.py`, `src/codeintel/build/state.py`
- Planning: `src/codeintel/build/hamilton/planner.py`
- Observability: `src/codeintel/build/hamilton/observability.py`

### Proposed end state
- One canonical representation:
  - A single `BuildPlan`/`PlanEntry` model with statuses + reasons + hashes.
- State and observability helpers become thin views over the plan (or are merged into the same
  module).

### Implementation steps
1. Decide the single source of truth (recommended: planner model, because it is Hamilton-first).
2. Refactor state and observability utilities to delegate to this model.
3. Update CLI handlers to use the unified plan/state output.
4. Add regression tests around:
   - “missing/stale/current/blocked” classification
   - staleness explanation (`dep_hashes` diffs)

### Acceptance gates
- CLI output is consistent across “status”, “plan”, and “why”.
- No duplication of hash/manifest logic across modules.

---

## Workstream G — Export consolidation (DAG-first export behavior)

### Goal
Avoid “two export systems” drifting:
- procedural exports under `src/codeintel/build/exports/*`
- Hamilton export targets under `src/codeintel/build/hamilton/native/export/*`

### Options (choose one)
1. **Hamilton targets are canonical**: CLI triggers DAG execution of export targets.
2. **Shared export engine**: keep `build/exports` as the only implementation of export logic, but
   Hamilton export targets call into it (so there is one writer).

### Implementation steps
1. Pick the canonical approach above.
2. Route all “export entrypoints” through that approach.
3. Delete the unused path (and update docs).
4. Add guardrails ensuring there is only one supported export invocation path.

### Acceptance gates
- Exports behave identically regardless of entrypoint.
- Only one implementation of “write JSONL/Parquet” remains.

---

## Workstream H — Reduce complexity in `support_factory` via extracted internals + tighter seams

### Goal
Lower maintenance cost of dynamic node generation while keeping determinism and typing strength.

### Current state
`src/codeintel/build/hamilton/nodes/support_factory.py` contains multiple concerns:
- signature mutation
- module mutation
- mapping construction
- node factory logic (datasets/loaders/artifacts/stubs)

### Proposed end state
- `support_factory` becomes orchestration; internal helpers hold the mechanics.
- Clear seams exist for:
  - signature attachment
  - node attachment
  - mapping/index building
  - table_key validation/parsing (single shared contract)

### Implementation steps
1. Extract internal helpers (e.g., `_signature_tools.py`, `_module_attach.py`, `_mappings.py`).
2. Add focused unit tests for each helper and a small integration test for module generation.
3. Ensure table_key validation/parsing uses the shared contract everywhere.
4. Remove any now-unused duplicated helper logic.

### Acceptance gates
- Smaller `support_factory.py`
- Deterministic generated module output
- Stable, test-covered internals

---

## Sequencing (recommended)

### Phase 0 — Inventory + guardrails scaffolding
- Add/extend fast guardrails that will protect refactors from regressions:
  - tagging API enforcement
  - materialization boundary enforcement
  - canonical entrypoint imports
- Add minimal scan tests under `tests/` for invariants.

### Phase 1 — Options loading unification (Workstream E)
- Low behavioral risk; improves correctness of planning/skip.

### Phase 2 — Tagging/naming unification (Workstream C)
- Mostly mechanical; reduces drift and enables better discovery.

### Phase 3 — Canonical patterns rollout (Workstream B)
- Refactor native targets incrementally domain-by-domain:
  - ingestion → graphs → analytics → export
- Delete bespoke glue once each module is migrated.

### Phase 4 — Target system entrypoint consolidation (Workstream A)
- After patterns stabilize, delete duplicate registries/catalogs to reduce confusion.

### Phase 5 — Plan/state/observability unification (Workstream F)
- Consolidate models and simplify CLI surfaces.

### Phase 6 — Export consolidation (Workstream G)
- Choose canonical export path and delete the other.

### Phase 7 — support_factory refactor (Workstream H)
- Safer after the rest is stable; mostly internal re-organization + tests.

---

## Validation / Quality Gates (run after each phase)
Use the repo’s consolidated quality gates:

```bash
scripts/bootstrap_codex.sh
uv sync
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

For fast iteration, also run focused tests/guardrails:

```bash
uv run python -m tools.guardrails
uv run pytest -q tests/test_build_storage_architecture_invariants.py
```

---

## Legacy Code Decommissioning Checklist
For every workstream that introduces a new canonical surface:
- Update all call sites to the canonical surface.
- Add a guardrail banning the old import/pattern.
- Delete the legacy module or reduce it to a 1–2 function forwarder temporarily.
- Delete the forwarder once downstream references are gone.

This prevents “half-migrated” ambiguity, which is particularly costly in DAG-centric systems.
