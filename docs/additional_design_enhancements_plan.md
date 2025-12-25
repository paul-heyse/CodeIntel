---
title: "Additional Design Enhancements Implementation Plan"
status: "design"
scope: "best-in-class hardening and DAG-first ergonomics"
related:
  - docs/build_test_failures_remediation_plan.md
  - docs/full_dag_basis_implementation_plan.md
  - docs/analytics_dag_migration_plan.md
---

# Additional Design Enhancements Implementation Plan

This plan operationalizes the five design ideas identified after Phase 1-7 of the
build remediation effort. The focus is on strengthening determinism, improving
extensibility, and reducing long-term maintenance risk while staying aligned with
the DAG-first target state.

## 1) Goals

- Make CLI configuration deterministic and framework-independent.
- Make node tagging enforceable, typed, and self-documenting.
- Move all data access into explicit DAG nodes and shared loader patterns.
- Expose schema inference failures as first-class, queryable artifacts.
- Add an auditable build decision trace for reproducibility and debugging.

## 2) Non-goals

- No compatibility or deprecation layers.
- No changes to public APIs unrelated to build/CLI behavior.
- No broad refactors outside the scope of the workstreams below.

## 3) Guiding principles

- DAG is the single source of truth for dependencies and metadata.
- Determinism over convenience (no implicit defaults without explicit metadata).
- Shared patterns over bespoke implementations.
- Diagnostics are data, not just logs.
- Every structural change comes with a validation gate.

## 4) Workstreams

### 4.1 CLI option metadata registry

**Objective:** Make env var names and option metadata explicit, stable, and shared.

**Key ideas**
- Introduce a small registry module that defines canonical option metadata.
- CLI command modules reference the registry instead of hand-built `Parameter` objects.
- Env var names are computed deterministically and rendered explicitly.

**Proposed modules**
- `src/codeintel/cli/options/registry.py`
- `src/codeintel/cli/options/types.py`

**Key types**
- `OptionSpec`: name(s), env_var, help, show_default, show_choices, group, etc.
- `OptionGroup`: shared bundles (e.g., runtime flags, output flags, verbosity).

**Primary changes**
- Move shared flags from `src/codeintel/cli/commands/_common.py` into the registry.
- Replace inline `Parameter(...)` definitions in `src/codeintel/cli/commands/*` with
  references to `OptionSpec` instances.
- Explicitly assign `env_var` for every CLI option, including nested `SharedFlags`.

**Acceptance**
- CLI help output uses explicit env vars for all options.
- `tests/build/hamilton/test_cli_snapshots.py` and
  `tests/build/hamilton/test_cli_env_vars.py` pass without snapshot drift.

---

### 4.2 Typed node tagging (NodeType + TagSpec)

**Objective:** Make node tagging typed and centrally enforced.

**Key ideas**
- Introduce a `NodeType` enum and a `TagSpec` dataclass.
- Require a `TagSpec` when attaching or generating nodes.
- Provide validation utilities to catch missing or invalid tags in one place.

**Proposed modules**
- `src/codeintel/build/hamilton/tag_spec.py`
- `src/codeintel/build/hamilton/tagging.py` (refactor)
- `src/codeintel/build/hamilton/nodes/module_attach.py` (update helpers)

**TagSpec fields**
- `node_type: NodeType`
- `domain: str`
- `target: str | None`
- `table_key: str | None`
- `artifact_name: str | None`
- `extra_tags: Mapping[str, str]`

**Primary changes**
- Add `TagSpec.from_context(...)` helpers for loaders/savers/materializations.
- Update `tagged_attach_node` to require a `TagSpec`.
- Add `validate_tag_spec` utilities to enforce `node_type` and required tags.

**Acceptance**
- `test_pr64_all_nodes_have_node_type_tag` passes.
- New validation tests assert TagSpec enforcement for loaders/savers.

---

### 4.3 DAG-first data access patterns

**Objective:** Move all data access into shared loader nodes and remove direct
`env.gateway` usage inside compute nodes.

**Key ideas**
- Expand loader patterns (`load_table`, `load_query`) with typed specs.
- Introduce a `DataAccessSpec` to describe source tables and SQL in a single place.
- Provide helper decorators for compute nodes to depend on loaders instead of
  directly pulling from the gateway.

**Proposed modules**
- `src/codeintel/build/hamilton/native/patterns/loaders.py` (extend)
- `src/codeintel/build/hamilton/native/patterns/access.py` (new)

**Primary changes**
- Create `load_table_spec(...)` and `load_query_spec(...)` helpers.
- Refactor analytics/graphs modules to consume loader nodes.
- Add a migration checklist for modules still using `env.gateway`.

**Acceptance**
- No direct `env.gateway` access inside native analytics/graphs compute nodes.
- Loader nodes expose consistent tagging (`node_type=loader_*`, `table_key`).

---

### 4.4 Schema inference errors as artifacts

**Objective:** Persist inference errors as data for observability and tooling.

**Key ideas**
- Record schema inference failures into a dedicated table or artifact.
- Keep inference errors non-fatal but durable across runs.

**Proposed schema**
- Table key: `core.schema_inference_errors`
- Row fields: `table_key`, `repo`, `commit`, `error`, `occurred_at`, `run_id`

**Primary changes**
- Extend `SchemaIndex` to expose inference error rows.
- Add a saver node to persist errors at end of schema compile.
- Register the table in schema registry and add a contract.

**Acceptance**
- Inference errors are written on failure and visible in storage.
- `test_schema_index_overrides` still passes and inference remains non-fatal.

---

### 4.5 Build decision trace artifact

**Objective:** Provide a deterministic, queryable audit trail for build decisions.

**Key ideas**
- Create a JSON artifact with per-target decisions (skip reason, input hash,
  options hash, inference status, provenance source).
- Generate once per build run and attach to build artifacts.

**Proposed artifact**
- Name: `build_decision_trace`
- Path template: `{build_dir}/build/decision_trace.json`
- Schema: JSON list of records with stable field order.

**Primary changes**
- Add a build-run collector node that assembles decision records.
- Persist as a file artifact using existing artifact saver patterns.
- Add a CLI subcommand to print or export the trace.

**Acceptance**
- Build runs produce the decision trace artifact deterministically.
- A small unit test validates schema and stable ordering.

## 5) Phased implementation plan

### Phase 0: Discovery and inventory
1) Inventory current CLI options and map them to command paths.
2) Enumerate all node types and existing tag keys.
3) Identify all direct `env.gateway` usages in analytics/graphs nodes.
4) Define schema for inference errors and build decision trace artifact.

Acceptance:
- Inventory artifacts checked in under `docs/_drafts/` or removed after use.

### Phase 1: CLI registry and explicit env vars
1) Implement `OptionSpec` and `OptionGroup` in `cli/options`.
2) Replace shared flags in `commands/_common.py` with registry usage.
3) Apply explicit `env_var` to all CLI options.
4) Update CLI help snapshots if needed.

Acceptance:
- CLI snapshot tests pass with no drift from framework defaults.

### Phase 2: TagSpec + NodeType
1) Add `NodeType` enum with canonical values.
2) Introduce `TagSpec` and validation utilities.
3) Update `tagged_attach_node` and loader/saver helpers to require TagSpec.
4) Add a small test suite for TagSpec validation.

Acceptance:
- All node tags validated centrally; missing tags fail fast.

### Phase 3: DAG-first loader refactor
1) Add `DataAccessSpec` and loader spec helpers.
2) Refactor analytics and graphs modules to use loaders.
3) Add a static check (or test) to flag `env.gateway` usage in nodes.

Acceptance:
- No direct gateway access inside compute nodes.

### Phase 4: Schema inference errors artifact
1) Add schema + contract for `core.schema_inference_errors`.
2) Extend schema compile to collect and persist error rows.
3) Add tests for inference error persistence.

Acceptance:
- Inference errors are captured as data without failing builds.

### Phase 5: Build decision trace artifact
1) Define JSON schema and output ordering.
2) Create collector node in build execution pipeline.
3) Persist as artifact and add a CLI view/export.
4) Add tests for deterministic output.

Acceptance:
- Decision trace artifact is generated and stable across runs.

## 6) Validation plan

- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- Run targeted tests by phase:
  - CLI: `tests/build/hamilton/test_cli_env_vars.py`,
    `tests/build/hamilton/test_cli_snapshots.py`
  - Tagging: `tests/build/hamilton/test_pr64_all_nodes_have_node_type_tag.py`
  - Schema: `tests/build/hamilton/test_pr72_manifest_v2.py`,
    `tests/build/hamilton/test_pr80_schema_*`
  - New tests added for TagSpec, inference errors, decision trace.

## 7) Deliverables

- New CLI options registry and explicit env var mapping.
- Typed tagging model (NodeType + TagSpec) with validation.
- Loader refactors for analytics/graphs nodes.
- Persisted schema inference errors dataset.
- Build decision trace artifact and CLI surface.

## 8) Risks and mitigations

- **Risk:** TagSpec migration breaks existing nodes.  
  **Mitigation:** provide a transitional adapter for auto-tagging in internal helpers.

- **Risk:** Loader refactors change runtime execution order.  
  **Mitigation:** validate DAG outputs and compare target inventories before/after.

- **Risk:** Decision trace grows too large.  
  **Mitigation:** cap per-target detail, store hashes instead of full payloads.

## 9) Open questions

- Should schema inference errors be a table, an artifact, or both?
- Should decision trace data be exposed via the serving API?
- Do we need a strict mode that fails on missing TagSpec instead of warning?
