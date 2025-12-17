<!--
This document is intentionally detailed and implementation-oriented.
It is a follow-on plan to deepen Hamilton (Apache) usage and make the DAG the
single source of truth across build, schemas, serving, and asset tracking.
-->

# Hamilton-First Intensification Plan (Follow-On)

> **Status**: Proposed implementation plan (ready to execute)  
> **Author**: AI Assistant  
> **Date**: 2025-12-17  
> **Audience**: Build/infra engineers and agents working in `src/codeintel/build/`  
> **Scope**: Deepen Hamilton-first architecture beyond current consolidation work

---

## Executive Summary

This plan intensifies the Hamilton-first architecture so the **Hamilton DAG becomes the single
source of truth** for:

- Target dependencies & execution plans
- Declared outputs (datasets + artifacts)
- IO boundaries (loaders/savers/materializers)
- Build artifacts (schema manifest, buildspec, semantic registry)
- Asset tracking/lineage emission
- Caching policy + parallel execution policy
- Observability (telemetry + DAG exports)

This is intentionally “breaking-change friendly”: the goal is a best-in-class integrated
Hamilton + Apache-based system with minimal duplicated infrastructure.

---

## Relationship to Existing Plans

- `docs/cleanup_post_hamilton_integration/BUILD_CONSOLIDATION_REMAINING_SCOPE.md`
  focuses on consolidation (templates/subdags, caching, context simplification, registry/schema
  unification, and parallel execution).
- This document assumes that work continues, but also introduces additional “Hamilton-hardening”
  scope that pushes further toward DAG-first correctness (explicit dependencies, IO enforcement,
  and DAG-derived emitted artifacts).

Where there is overlap (e.g., parallelism, caching, registry unification), the **Hamilton-first**
version here should be treated as the end-state.

---

## North Star (Non-Negotiables)

1. **No hidden dependencies**: all data reads/writes used to compute outputs are expressed as
   Hamilton nodes and appear in the DAG.
2. **No parallel registries**: one canonical representation of targets/datasets/artifacts is
   derived from the DAG and enriched with metadata.
3. **No hidden IO**: writes are explicit saver/materializer nodes; reads come from loader nodes.
4. **Deterministic, inspectable builds**: build outputs and emitted artifacts are reproducible
   from the DAG + snapshot inputs.
5. **Operational hardness**: invariants are enforced by validators/tests, not conventions.

---

## Acceptance Gates (Per PR + Per Sprint)

For every PR:
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- `uv run pytest -q`

For each sprint milestone:
- `codeintel` CLI help/rendering tests remain stable (no regressions).
- DAG validation gate passes (Phase 0 introduces/extends this).

---

## Terminology & Node Taxonomy (Canonical)

Use tags and naming conventions consistently:

- **Targets**: `t__<target>` nodes tagged `node_type="materialize"` (exactly one per target)
- **Compute**: `node_type="compute"` (pure computations; no storage IO)
- **Loaders**:
  - `node_type="loader.query"` returns Ibis expressions (e.g., `q__analytics__function_metrics`)
  - `node_type="loader.dataframe"` returns executed/pandas data (e.g., `df__...`)
- **Datasets**: `node_type="dataset"` nodes tagged with `table_key`
- **Artifacts**: `node_type="artifact"` nodes tagged with `artifact`
- **Helpers**: `node_type="helper"` intermediate DAG nodes that should not be treated as
  target completion boundaries

This taxonomy is enforced by validation in Phase 0.

---

## Phase Plan (Comprehensive)

### Phase 0 — Hamilton DAG Contract + Validator Upgrade (1–3 days)

**Objective**: Convert the “Hamilton-first” architecture into enforceable invariants.

**Key work items**
- Extend `src/codeintel/build/hamilton/validate.py` to enforce:
  - Exactly one `node_type="materialize"` node per target.
  - Dataset nodes have `table_key` and a single producing target.
  - Artifact nodes have `artifact` and a single producing target.
  - Output contracts match DAG-derived outputs (tables/artifacts).
  - Compute nodes are “IO-pure” (no `env.gateway.ibis.table(...)`, no `.execute()`).
- Add a CLI entrypoint (or build command) to run validation and emit JSON for CI usage.
- Add a small golden “DAG contract” doc section for developers (this file).

**Deliverables**
- Stronger validator + CI-friendly output.
- A small test suite that fails fast when invariants are violated.

**Acceptance**
- Validator passes on current DAG.
- Introducing a second materialize node for a target fails validation.

---

### Phase 1 — DAG-Explicit Storage Dependencies (1–2 weeks)

**Objective**: Eliminate hidden reads from compute nodes. All upstream data flows through
Hamilton loader nodes.

**Rules**
- `node_type="compute"` nodes MUST NOT call:
  - `env.gateway.ibis.table(...)`
  - `.execute()` on Ibis expressions
  - direct SQL execution helpers
- If executed rows are needed, consume a `df__...` loader node.
- If Ibis expressions are needed, consume a `q__...` loader node.

**Key work items**
- Add/upgrade loader nodes in templates for:
  - `q__<schema>__<table>` (Ibis query loader)
  - `df__<schema>__<table>` (DataFrame loader)
  - `a__<artifact>` (artifact loader where applicable)
- Refactor native modules to take explicit loader inputs instead of reading from gateway.
- Add a validator/test that walks the built Hamilton nodes, inspects origin functions, and
  fails on forbidden patterns (string/AST scan of `inspect.getsource` is acceptable).

**Deliverables**
- A build system where target closure planning is correct because the DAG expresses real deps.
- Deterministic dependency hashing: input hashes can be computed from explicit deps.

**Acceptance**
- Validator enforces “compute nodes are IO-pure”.
- Target execution plans accurately reflect real data dependencies (no implicit reads).

---

### Phase 2 — DAG-First Target Registry & Planning Everywhere (3–5 days)

**Objective**: Make every planning, schema, and serving pipeline consume the DAG-derived view.

**Key work items**
- Define a single “derived registry” concept:
  - Build driver → derive deps/outputs from DAG → enrich with `OutputTarget` metadata.
- Migrate consumers to use this derived registry:
  - build executor planning
  - schema provider compilation paths
  - CLI “plan” and “list targets” commands
- Tighten strictness in critical paths (fail fast when the DAG violates the contract).

**Deliverables**
- “One graph to rule them all”: no divergent dependency sources.

**Acceptance**
- Removing static `dependencies` does not break planning (or the field becomes non-authoritative).

---

### Phase 3 — Template/SubDAG Expansion + Native Module Thinning (2–3 weeks)

**Objective**: Drive repetition down by pushing common target shapes into reusable
`@subdag`/`@parameterized_subdag` templates.

**Key work items**
- Expand the template library beyond current templates:
  - Table pipelines (Ibis → saver → TargetRunRecord)
  - Row pipelines (rows → saver → TargetRunRecord)
  - Tool/artifact pipelines (artifact → saver → TargetRunRecord)
  - Domain templates for repeated extraction/metrics patterns
- Convert targets to:
  - a minimal native override module that defines only unique compute logic
  - plus a template binding for IO/materialization
- Expand `@pipe_input` usage for multi-step transforms so intermediate steps are DAG-visible.

**Deliverables**
- “Thin native overrides” architecture with fewer moving parts.

**Acceptance**
- Reduced duplication and smaller native surface area without changing produced outputs.

---

### Phase 4 — DAG-Visible Emitted Artifacts (Serving + Asset Catalog) (1–2 weeks)

**Objective**: Make emitted build artifacts first-class DAG outputs.

**Primary artifacts to make DAG outputs**
- `schema_manifest.json`
- `buildspec.json`
- `semantic_registry.json`
- Asset catalog materialization/lineage emission (see `src/codeintel/build/assets/emitter.py`)

**Key work items**
- Choose an emission architecture:
  - **Option A (hook)**: A Hamilton lifecycle adapter collects `TargetRunRecord`s at the end of
    a run and calls `persist_asset_catalog_for_run(...)`.
  - **Option B (target)**: Create a dedicated `asset_catalog` target that materializes rows into
    tracking tables, so lineage is a target output and can be planned/re-run.
  - (Option B is the purest DAG-first approach; Option A is simpler operationally.)
- Represent serving artifacts as artifact nodes written by materializers, so “publish serving
  snapshot” is driven by DAG outputs rather than ad-hoc orchestration.
- Ensure emitted artifacts are stable and include the DAG-derived metadata (deps/outputs/versions).

**Deliverables**
- Serving snapshots are reproducible and traceable.
- Asset lineage is consistent and derived from the same DAG used to execute targets.

**Acceptance**
- A build run can produce all serving artifacts via explicit target execution.
- Asset catalog emission is deterministic and validated.

---

### Phase 5 — First-Class Caching + Versioning (~1 week)

**Objective**: Treat caching as an explicit, observable execution mode with strong defaults.

**Key work items**
- Standardize `.with_cache(...)` configuration and storage location under build dir.
- Expand `@cache` usage only on deterministic pure-Python nodes (parsing, indexing, enumeration).
- Expose cache controls in CLI/config:
  - enable/disable
  - clear
  - report cache hit rate and per-node behaviors

**Acceptance**
- Cache improves iterative runs without correctness risk (no caching of Ibis expressions/data writes).

---

### Phase 6 — Parallel Execution as a First-Class Mode (~1 week)

**Objective**: Make parallel execution a supported mode with correctness guarantees.

**Key work items**
- Promote threadpool execution behind stable CLI/config settings.
- Use node tags to gate locks and ordering:
  - global write lock for `node_type="materialize"`
  - optional per-target max parallelism from resources/metadata
- Add determinism tests (sequential vs threadpool) for a representative subset of targets.

**Acceptance**
- Threadpool mode is safe by default and produces identical manifests/outputs.

---

### Phase 7 — Observability & Developer UX (3–5 days, ongoing)

**Objective**: Make the DAG explorable and “debuggable by default”.

**Key work items**
- Add “export DAG” tooling:
  - JSON export of full Hamilton graph + derived target graph
  - DOT/Mermaid export for human review
- Add “why” tooling:
  - explain why a target depends on another target (path explanation)
  - show which datasets/artifacts connect them
- Standardize tags for UI/telemetry and keep them stable.

**Acceptance**
- One-command introspection for plans, deps, and outputs that matches runtime reality.

---

## Concrete Sprint Plan (Recommended)

Assume 2-week sprints; adjust as needed. This sequence is designed to deliver correctness early
and reduce refactor risk.

### Sprint 1 — DAG Contract Hardening + Guardrails

**Goals**
- Make “Hamilton-first correctness” enforceable before large refactors land.

**Scope**
- Phase 0 fully.
- Start Phase 1 by adding the enforcement test (even if refactors come later).

**Deliverables**
- Upgraded DAG validator + CLI command to run it.
- Forbidden-pattern enforcement (compute nodes cannot do IO).
- Documented node taxonomy and tags.

**Exit criteria**
- Validator + tests pass in CI; enforcement catches a seeded violation.

---

### Sprint 2 — Explicit Dependencies (Ingestion + Graphs)

**Goals**
- Remove the highest-value hidden deps first (foundation layers).

**Scope**
- Phase 1 for ingestion and graph targets:
  - Replace gateway reads inside compute with `q__...` and `df__...` inputs.
  - Ensure loader nodes exist for required tables/artifacts.

**Deliverables**
- Ingestion/graphs native modules consume explicit loader nodes only.
- Closure planning becomes correct for these domains.

**Exit criteria**
- Validator passes with compute-IO purity.
- No ingestion/graphs compute node reads storage directly.

---

### Sprint 3 — Explicit Dependencies (Analytics + Export) + Registry Unification

**Goals**
- Complete DAG-explicit dependency work and ensure all planners use the DAG.

**Scope**
- Finish Phase 1 for analytics/export.
- Phase 2: migrate remaining consumers to DAG-derived registry.

**Deliverables**
- All compute nodes across all domains are IO-pure.
- Build planning, schema compilation, and CLI planning use derived deps/outputs.

**Exit criteria**
- Planning outputs match actual execution dependencies.
- Removing a static dependency does not change execution order (DAG is authoritative).

---

### Sprint 4 — Template Expansion + Native Thinning

**Goals**
- Reduce long-term maintenance cost by consolidating repeated target shapes.

**Scope**
- Phase 3:
  - Expand template library where repetition remains.
  - Convert a first large batch of targets to `@parameterized_subdag`.
  - Expand `@pipe_input` for complex transforms to improve DAG visibility.

**Deliverables**
- Native modules become thin overrides.
- Measurable reduction in duplicated boilerplate across targets.

**Exit criteria**
- Target outputs unchanged; code footprint reduced; DAG readability improved.

---

### Sprint 5 — DAG-Visible Emitted Artifacts (Serving + Asset Catalog)

**Goals**
- Ensure all “meta outputs” are produced from the DAG, not ad-hoc code.

**Scope**
- Phase 4:
  - Implement either hook-based or target-based asset catalog emission.
  - Make `buildspec.json` and `schema_manifest.json` first-class artifacts.
  - Ensure serving publish consumes those artifacts deterministically.

**Deliverables**
- Reproducible serving snapshot artifacts.
- Asset versions and lineage emitted from a single canonical pipeline.

**Exit criteria**
- A build run can emit serving artifacts and asset catalog data via explicit DAG steps.

---

### Sprint 6 — Caching + Parallelism Productionization

**Goals**
- Make caching and parallel execution first-class, observable, and safe.

**Scope**
- Phase 5 + Phase 6:
  - Standard cache configuration and CLI toggles.
  - Expand `@cache` on deterministic pure-Python nodes.
  - Promote threadpool execution and add determinism tests.

**Deliverables**
- Caching gives tangible speedups on iterative runs.
- Parallel mode is safe-by-default and validated.

**Exit criteria**
- Determinism tests pass; parallel mode enabled for safe target classes.

---

### Sprint 7 — Observability & “DAG UX”

**Goals**
- Make it easy to understand “why” and “what changed” in DAG terms.

**Scope**
- Phase 7:
  - DAG export tools and CLI commands.
  - Dependency explanation tooling.
  - Telemetry summarization for per-run reporting.

**Deliverables**
- Best-in-class developer experience for introspection and debugging.

**Exit criteria**
- One-command DAG export and dependency explanation for any target.

---

## Recommended PR Slicing Strategy

To keep diffs reviewable and risk low:

1. **PRs that only add validators/tests** (no behavior changes).
2. **Refactor PRs by domain** (ingestion → graphs → analytics → export).
3. **Template migrations in batches** (e.g., convert 3–5 targets per PR).
4. **Artifact emission changes behind flags** initially, then make canonical.
5. **Parallelism/caching toggles** shipped early but defaulted conservatively.

---

## Risk Register (What to Watch)

- **Runtime type-hint evaluation**: Hamilton evaluates annotations at runtime.
  Any types used in node signatures must be importable at runtime.
- **Implicit reads**: most correctness bugs come from hidden reads not expressed in the DAG.
- **Determinism under parallelism**: materialization locking and run ordering must be explicit.
- **Schema drift**: schema provider needs to align with DAG-derived outputs and emitted artifacts.

---

## Success Metrics (End-State)

- Hamilton DAG is the single source of truth for deps/outputs/plans.
- Compute nodes are IO-pure; all reads/writes are explicit DAG nodes.
- Emitted artifacts (buildspec/schema manifest/semantic registry/asset catalog) are DAG outputs.
- Native code surface area reduced; templates/subdags are the default.
- Caching and parallelism are first-class modes with deterministic behavior and clear telemetry.

