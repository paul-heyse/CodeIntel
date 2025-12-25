---
title: "Build Test Failures Remediation Plan"
status: "design"
scope: "pytest failures + DAG-first hardening"
related:
  - docs/full_dag_basis_implementation_plan.md
---

# Build Test Failures Remediation Plan

This plan addresses the current pytest failures (build scope) and extends the
architecture toward the DAG-first target state described in
`docs/full_dag_basis_implementation_plan.md`. The intent is to resolve the
immediate test failures while improving correctness, clarity, extensibility, and
maintainability of the build system.

## 1) Goals

- Eliminate all failing tests and reduce the likelihood of regression.
- Strengthen the DAG as the sole source of truth for dependencies, I/O, and
  schema provenance.
- Improve API clarity and enforce consistent contracts across build and storage.
- Make instrumentation (tags, manifests, provenance) deterministic and
  verifiable.

## 2) Non-goals

- No compatibility or deprecation scaffolding.
- No refactors unrelated to the failing areas or DAG-first scope.

## 3) Failure clusters and target outcomes

### 3.1 Skip logic signature drift
**Failure:** `TestSkipCheckRequest.test_skip_check_no_manifest_returns_false`  
**Outcome:** `should_skip` uses a manifest interface with keyword-only args.

### 3.2 CLI snapshot drift (env var names)
**Failure:** 26 `test_cli_snapshots` mismatches  
**Outcome:** CLI env var names are stable and explicit; snapshots only change
when intended.

### 3.3 Missing `node_type` tags
**Failure:** `test_pr64_all_nodes_have_node_type_tag`  
**Outcome:** All non-generated nodes carry canonical `node_type` tags.

### 3.4 Schema provenance and inference errors
**Failure:** `test_pr72_manifest_v2`, `test_pr80_*`, `test_schema_index_overrides`  
**Outcome:** Provenance is complete; inference errors are recorded, not fatal.

### 3.5 Registry inconsistencies
**Failure:** `test_all_targets_have_tables`, `test_all_targets_have_no_static_dependencies`  
**Outcome:** Artifact-only targets are supported; static dependencies are zeroed.

### 3.6 DuckDB ambiguity in serving publisher
**Failure:** Binder ambiguity on `build` schema  
**Outcome:** All SQL uses fully-qualified references or avoids ambiguous schema.

### 3.7 Node naming invalid identifiers
**Failure:** `test_node_names_are_valid_identifiers`  
**Outcome:** Generated pipeline node names are sanitized.

### 3.8 Export test table collisions
**Failure:** `test_build_export_relation_uses_storage_export_service`  
**Outcome:** schema seeding avoids collisions or uses CREATE IF NOT EXISTS.

## 4) Architectural principles (aligned with DAG-first target state)

- **Single source of truth:** DAG-derived contracts and metadata override static
  definitions.
- **Explicit boundaries:** read/write nodes, manifest access, and schema
  inference are explicit, typed, and tested.
- **Deterministic metadata:** tags and provenance are stable and never inferred
  from mutable global state.
- **Composable templates:** shared patterns for loaders, savers, and collectors.

## 5) Phased implementation plan

### Phase 0: Safety and diagnostics (setup)
1) Add a short-lived diagnostics file to capture:
   - Node names lacking tags.
   - DAG nodes with invalid identifiers.
   - Target outputs with missing schemas.
2) Confirm failing tests map to the clusters in Section 3.

Acceptance:
- A minimal diagnostic summary exists for the failures in this plan.

---

### Phase 1: Manifest interface and skip logic hardening
**Scope:** `src/codeintel/build/hamilton/run_records.py` (and related)

1) Introduce a `BuildManifestService` Protocol:
   - `load_manifest(*, target: str, repo: str, commit: str) -> OutputManifest | None`
   - Keyword-only signature to enforce correct usage.
2) Update `should_skip` to call `load_manifest` via keyword arguments.
3) Provide a small adapter on `BuildEnv` to expose the manifest service.
4) Update tests and mocks to use the protocol (no positional args).

Acceptance:
- `TestSkipCheckRequest` passes.
- Any existing callers compile without positional-argument usage.

---

### Phase 2: Canonical node tagging and loader patterns
**Scope:** `src/codeintel/build/hamilton/tagging.py`,
`src/codeintel/build/hamilton/nodes/module_attach.py`,
`src/codeintel/build/hamilton/native/patterns/`

1) Add `tagged_attach_node(...)` helper that:
   - Requires `node_type` or derives it from the helper context.
   - Preserves existing tags and merges explicit `extra_tags`.
2) Create loader helpers:
   - `load_table(domain, target, table_key, node_name=...) -> ir.Table`
   - `load_query(domain, target, sql, node_name=...) -> ir.Table`
   - Tags include `node_type=loader_*` and `table_key`.
3) Refactor hotspot (and other analytics/graphs modules) to use loader helpers
   instead of reading directly from `env.gateway`.

Acceptance:
- `test_pr64_all_nodes_have_node_type_tag` passes.
- `test_schema_registry_consumers` recognizes hotspots as a consumer of
  `core.modules` via tags.

---

### Phase 3: Schema provenance and inference stability
**Scope:** `src/codeintel/build/schemas/compile.py`,
`src/codeintel/build/schemas/schema_index.py`,
`src/codeintel/build/schemas/provider_declared.py`

1) Adjust provenance collection so artifact provenance is always a dict in
   v2 manifests, even when empty.
2) When compiling schema manifests for inferred tables:
   - First consult DAG-derived outputs and the SchemaIndex cache.
   - Use the declared provider only as fallback.
3) Ensure inference errors are recorded in SchemaIndex without raising, when
   inference is the only source for a schema.
4) Preserve deterministic ordering and stable output.

Acceptance:
- `test_pr72_manifest_v2` passes.
- `test_pr80_schema_compile_uses_batch_inference` and
  `test_pr80_schema_manifest_identical_batch_vs_individual` pass.
- `test_schema_index_overrides` passes.

---

### Phase 4: Registry normalization (artifact-only targets and static deps)
**Scope:** `src/codeintel/build/targets.py`,
`src/codeintel/build/target_spec_compiler.py`,
`src/codeintel/build/target_catalog.py`

1) Treat artifact-only targets as valid outputs:
   - Allow empty table lists when artifacts exist.
   - Validate in registry checks.
2) Ensure static dependencies are zeroed in compiled OutputTargets:
   - Enforce `dependencies=()` in spec compilation.
   - Store dependencies only in DAG-derived metadata.

Acceptance:
- `test_all_targets_have_tables` passes (including scip_proto).
- `test_all_targets_have_no_static_dependencies` passes.

---

### Phase 5: DuckDB ambiguity and test seeding
**Scope:** `src/codeintel/build/serving/publisher.py`,
`src/codeintel/storage/serving/search_index.py`,
`tests/_helpers/schemas.py`, `tests/build/exports/test_export_service_boundary.py`

1) Introduce a `fully_qualified_table_ref(...)` helper to generate explicit
   `main.schema.table` or `schema.table` references as needed.
2) Update storage SQL helpers to use the helper.
3) Ensure test schema seeding uses `CREATE TABLE IF NOT EXISTS` or avoids
   double-creation when `ensure_production_schemas` pre-creates tables.

Acceptance:
- Serving publisher tests pass without Binder errors.
- Export boundary test passes without table collision.

---

### Phase 6: Node naming policy for pipeline-generated nodes
**Scope:** `src/codeintel/build/hamilton/naming.py`,
`src/codeintel/build/hamilton/native/graphs/graph_targets.py`

1) Add a sanitizer for pipeline namespace components:
   - Convert `.` and other invalid characters to `_`.
2) Apply sanitizer in any `@pipe_input` or macro-generated node names.

Acceptance:
- `test_node_names_are_valid_identifiers` passes.

---

### Phase 7: CLI env var stability
**Scope:** `src/codeintel/cli/handlers/*`, shared CLI options

1) Make env var names explicit on CLI options to prevent framework defaults.
2) Add a unit test (or snapshot update gate) for env var name stability.

Acceptance:
- All CLI snapshot tests pass with stable env var names.

## 6) Implementation sequencing and dependencies

1) Phase 1 (manifest interface) unlocks skip logic stability.
2) Phase 2 (tags + loaders) unlocks consumer/producer registry correctness.
3) Phase 3 (provenance + inference) unlocks schema manifest tests.
4) Phase 4 (registry normalization) resolves table/dep registry tests.
5) Phase 5 (DuckDB ambiguity) resolves serving/export tests.
6) Phase 6 (naming) resolves identifier test.
7) Phase 7 (CLI env var stability) resolves snapshot suite.

## 7) Validation plan

- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- Run `uv run codeintel build validate --output-format json`.
- Run targeted pytest scopes in order:
  - `tests/build/hamilton/native/test_skip_logic.py`
  - `tests/build/hamilton/test_pr64_all_nodes_have_node_type_tag.py`
  - `tests/build/hamilton/test_pr72_manifest_v2.py`
  - `tests/build/hamilton/test_pr80_schema_*`
  - `tests/build/test_registry*.py`
  - `tests/build/serving/test_publisher.py`
  - `tests/build/exports/test_export_service_boundary.py`
  - `tests/build/hamilton/test_cli_snapshots.py`

## 8) Deliverables (per phase)

- Code changes in relevant modules.
- Updated snapshots (only after env var stability is restored).
- A short validation note describing test scopes run and outcomes.

## 9) Risks and mitigations

- **Risk:** tag changes alter derived output inventory.  
  **Mitigation:** validate DAG and compare `compile_output_targets` output before/after.

- **Risk:** schema provenance changes impact manifest consumers.  
  **Mitigation:** keep v2 serialization stable and add coverage to schema manifests.

- **Risk:** loader refactors change compute signatures.  
  **Mitigation:** update `__all__` and expose loader nodes in module exports.

## 10) Success criteria

- All previously failing tests pass.
- DAG-derived catalogs are stable and match runtime outputs.
- No static dependency drift; all outputs are derived from DAG metadata.
