# Systemic Test Failures Remediation Plan

## 1) Goals
- Resolve the current test failures by addressing root causes rather than patching tests.
- Make validation/contract behavior stable, predictable, and DAG-first (lenient by default).
- Restore CLI shared flags correctness and env var stability.
- Align analytics row writers with schema registry definitions (Pandera-first).
- Enforce architecture boundaries (DuckDB usage in storage only) and improve tooling resilience.

## 2) Non-goals
- No compatibility shims or deprecation layers (single-developer, aggressive migration).
- No broad refactors outside the identified failure clusters.
- No changes to external tools beyond controlled wrappers (e.g., scip-python binary).

## 3) Scope Summary (Ordered)
1) Validation/contract strategy (lenient default)
2) Schema inference errors artifact
3) Shared CLI flags redesign
4) Data model alignment (analytics writers + goldens)
5) Boundaries and tooling (DuckDB localization, SCIP resiliency, performance)

## 4) Guiding Principles (DAG-first)
- Build artifacts and contracts are produced by the DAG, not inferred at runtime.
- Contract validation is explicit and scoped to validation commands or build phases.
- Schema registry is canonical; writers must conform to Pandera schemas.
- Storage layer owns DuckDB usage; higher layers consume storage services.
- CLI options are stable, explicit, and environment-variable deterministic.

## 5) Clarifying Specs and Derisking Details

### 5.1 Validation Mode Semantics
- `OFF`: no contract checks, no validation summary artifact, no warnings.
- `LENIENT` (default): missing tables are ignored; column mismatches are warnings; summary artifact is written.
- `STRICT`: missing tables, column mismatches, or missing contracts raise errors and stop execution.

Validation mode mapping:

| Context | Default Mode | Expected Behavior |
| --- | --- | --- |
| Gateway open (read-only or empty DB) | `LENIENT` | Log issues + summary artifact; no raises |
| `codeintel build run` (default) | `LENIENT` | Warn-only unless `--strict-contracts` or `--validation-mode required` |
| `codeintel build run --strict-contracts` | `STRICT` | Fail fast on any contract mismatch |
| `codeintel storage validate-macros` | `STRICT` | Fail fast with full issue list |
| `codeintel docs export --validation-mode required` | `STRICT` | Fail fast with validation errors |
| CI validation or explicit validation commands | `STRICT` | Fail fast |

### 5.2 Contract Validation Surfaces
- Order of operations:
  1) Project resolution (missing `codeintel.yaml` must fail before DB open).
  2) Storage open (lenient in default paths).
  3) Explicit validation (strict, only in validation commands or `--validation-mode required`).
- Exit codes:
  - Project/config validation failures map to `CLI_EXIT_VALIDATION`.
  - Contract violations in strict mode map to `CLI_EXIT_VALIDATION`.
  - Missing DB in read-only contexts maps to a storage error (non-validation exit).
- Required strict surfaces:
  - `codeintel storage validate-macros`
  - `codeintel docs export --validation-mode required`
  - `codeintel datasets lint/verify` (when validation requested)

### 5.3 Shared Flags Implementation Contract
- The shared flags field must be flattened per-command with a fresh `Parameter(name="*")`.
- Avoid reusing a global `Parameter` instance (Cyclopts mutates metadata).
- `SharedFlags` must be a concrete dataclass returned by `shared_flags_type()`.
- Shared flags must include `--root`, `--output-format`, `--json`, and `--verbose`.
- Env var names must be explicit and deterministic (`CODEINTEL_<COMMAND_PATH>_<ARG>`).

Reference pattern:

```python
_BUILD_RUN_FLAGS_FIELD = shared_flags_field(BUILD_RUN_PATH)

@dataclass
class BuildRunCommand:
    flags: SharedFlags = _BUILD_RUN_FLAGS_FIELD
```

### 5.4 Schema Inference Errors Artifact Spec
- Table key: `core.schema_inference_errors`
- Required columns: `table_key`, `repo`, `commit`, `error`, `occurred_at`, `run_id`
- Always materialize the table; empty rows when no errors exist.
- Include in build artifacts (schema manifest + serving artifacts).
- Validation should treat this dataset as canonical and always present in strict mode.

### 5.5 Data Model Alignment Targets
- `analytics.behavioral_coverage`:
  - Ensure required non-null columns are populated (`test_id`, `behavior_tags`, `tag_source`).
  - Remove `function_goid_h128` unless the schema is updated to include it.
- `core.modules`:
  - Decide on `row_hash` policy: computed deterministically or nullable with golden updates.
- General rule:
  - Writers must construct rows from `SCHEMA_REGISTRY` row models (no ad-hoc dicts).

### 5.6 Test Gating Order (Stop Conditions)
- Phase 1 gate: `tests/cli/test_build_command.py`, `tests/storage/test_conformance.py`
  - Stop if any `Dataset contract validation failed` errors remain.
- Phase 2 gate: `tests/config/test_dataset_contract_snapshot.py`
  - Stop if dataset count snapshots are unstable.
- Phase 3 gate: `tests/cli/test_help_rendering.py`, `tests/cli/test_cli_scope_and_plan.py`
  - Stop if shared flags are missing from help or plan.
- Phase 4 gate: `tests/analytics/test_profiles_and_functions.py`,
  `tests/analytics/test_graph_features.py`, `tests/storage/test_table_goldens.py`
  - Stop on any Pandera schema errors or golden diffs.
- Phase 5 gate: `tests/architecture/test_duckdb_boundaries.py`,
  `tests/ingestion/test_tools.py`, `tests/cli/performance/test_performance.py`
  - Stop if boundaries, tooling, or performance budgets fail.

## 6) Phased Implementation Plan

### Phase 1: Validation/Contract Strategy (Lenient by Default)

#### Objectives
- Prevent gateway open from failing on missing tables in empty or partial DBs.
- Ensure strict validation is still available and used where it matters.
- Align validation behavior with DAG-first outputs and explicit validation commands.

#### Implementation Tasks
- Add `ValidationMode` (e.g., `STRICT`, `LENIENT`, `OFF`) and default to `LENIENT`.
  - Update `StorageConfig` to include `validation_mode` with default `LENIENT`.
  - Add CLI flag `--validation-mode` where relevant (storage validate, docs export, build run).
- Modify contract validation to treat missing tables as unknown in lenient mode.
  - `TableColumnsLookup` should return `None` for non-existent tables.
  - `_validate_table_columns` should skip tables when lookup returns `None`.
- Split validation into two layers:
  - `validate_contract_or_raise_strict()` for explicit validation commands.
  - `collect_contract_issues_lenient()` for informative logging on gateway open.
- Persist a small validation summary artifact (for observability), not a hard gate.
- Add tests that verify:
  - Empty DBs open without contract failures in lenient mode.
  - Strict mode still fails with correct error messages.

#### Acceptance Criteria
- All tests previously failing on `Dataset contract validation failed` during setup pass.
- CLI exit codes and error paths match expectations (nonexistent `codeintel.yaml` is reported).
- Contract validation remains strict for explicit commands and CI validation paths.

#### Likely Touchpoints
- `src/codeintel/storage/gateway/config.py`
- `src/codeintel/storage/gateway/factory.py`
- `src/codeintel/storage/validation/contract.py`
- `src/codeintel/core/schemas/contract_validation.py`
- CLI validation entry points in `src/codeintel/cli/`

#### Phase 1 Tests
- `tests/cli/test_build_command.py`
- `tests/cli/test_datasets_command.py`
- `tests/cli/test_storage_command.py`
- `tests/storage/test_conformance.py`

---

### Phase 2: Schema Inference Errors Artifact

#### Objectives
- Make `core.schema_inference_errors` a first-class DAG artifact.
- Ensure the table exists for validation and is populated (or empty) deterministically.

#### Implementation Tasks
- Ensure table schema is included in the canonical schema provider and contract catalog.
- Add DAG node(s) that capture schema inference errors into rows for
  `core.schema_inference_errors`.
- Ensure materialization writes an empty table when no errors exist.
- Expose the artifact in build outputs and schema manifests for visibility.
- Update dataset contract snapshots to include the new dataset counts.

#### Acceptance Criteria
- No contract validation error referencing `schema_inference_errors`.
- Snapshot counts updated and stable in tests.
- Build artifact includes inference error table metadata.

#### Likely Touchpoints
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/build/hamilton/native/export/serving_artifacts.py`
- `src/codeintel/build/schemas/schema_index.py`
- Dataset snapshot tests under `tests/config/`

#### Phase 2 Tests
- `tests/config/test_dataset_contract_snapshot.py`
- `tests/build/hamilton/test_pr72_manifest_v2.py`

---

### Phase 3: Shared CLI Flags Redesign

#### Objectives
- Restore `--output-format`, `--root`, `--json`, `--verbose` in help and execution.
- Guarantee per-command env var stability with explicit `env_var=...`.

#### Implementation Tasks
- Redesign `shared_flags_field()` so Cyclopts sees a unique flattening Parameter per field:
  - Avoid reusing a shared `Parameter(name="*")` instance.
  - Prefer a per-command dataclass with `Annotated[..., Parameter(name="*")]` on the
    `flags` field or equivalent in Cyclopts.
- Make `SharedFlags` a concrete dataclass for flattening (not just a Protocol).
- Add explicit `env_var=` values in option metadata for shared flags.
- Ensure `ProjectResolver` runs before gateway open to produce expected errors
  (`codeintel.yaml` not found, etc.).
- Update CLI snapshots and help rendering tests if needed.

#### Acceptance Criteria
- CLI help displays output-format, root, and shared flags across commands.
- CLI plan accepts `--output-format` and returns plan output successfully.
- Exit codes align with tests (`expected 1, got 2` issues resolved).

#### Likely Touchpoints
- `src/codeintel/cli/options/shared_flags.py`
- `src/codeintel/cli/options/registry.py`
- `src/codeintel/cli/commands/*`
- `src/codeintel/cli/resolution/`
- CLI snapshot tests in `tests/cli/`

#### Phase 3 Tests
- `tests/cli/test_build_cli.py`
- `tests/cli/test_help_rendering.py`
- `tests/cli/test_cli_scope_and_plan.py`
- `tests/cli/test_typer_cli.py`
- `tests/build/hamilton/test_cli_snapshots.py`

---

### Phase 4: Data Model Alignment (Analytics Writers)

#### Objectives
- Ensure analytics writers emit rows that conform to schema registry definitions.
- Remove schema drift and stabilize goldens.

#### Implementation Tasks
- Align analytics row writers with `SCHEMA_REGISTRY` row models.
  - Provide row factory helpers for each dataset in analytics writers.
- Fix `analytics.behavioral_coverage` writer to supply required non-null columns
  and remove unexpected fields.
- Decide on `row_hash`:
  - Either compute it deterministically in writers, or keep nullable and update goldens.
- Make test schema setup idempotent (avoid table-exists failures).
  - Use `CREATE TABLE IF NOT EXISTS` or a dedicated test schema helper.
- Update golden files and dataset snapshot counts after schema alignment.

#### Acceptance Criteria
- No Pandera schema errors in analytics tests.
- Table golden matches expected output including any new columns.
- Graph feature tests use idempotent setup and pass.

#### Likely Touchpoints
- `src/codeintel/build/analytics/**`
- `src/codeintel/build/hamilton/contracts/schemas/`
- `tests/analytics/test_profiles_and_functions.py`
- `tests/analytics/test_graph_features.py`
- `tests/storage/test_table_goldens.py`

#### Phase 4 Tests
- `tests/analytics/test_profiles_and_functions.py`
- `tests/analytics/test_graph_features.py`
- `tests/storage/test_table_goldens.py`

---

### Phase 5: Boundaries and Tooling

#### Objectives
- Keep DuckDB usage inside storage modules only.
- Make SCIP tooling robust in tests and non-git contexts.
- Reduce read-path latency for CLI operations.
- Fix subprocess registry semantics.

#### Implementation Tasks
- Move DuckDB snapshot creation and search index build from
  `src/codeintel/build/serving/publisher.py` into a storage service module.
- Introduce a storage-level `ServingSnapshotService` used by build layer.
- Update tests enforcing DuckDB boundaries.
- Add a SCIP test stub or fallback when repo is not a git checkout.
  - Allow tool runner to operate in "no-git" mode with deterministic metadata.
- Clarify subprocess registry semantics:
  - `unregister_subprocess` removes entries, or provide `mark_exited` and keep
    `unregister` as removal.
- Add a CLI read-path cache or memoized DAG manifest loader to keep
  `build.status` under performance budget.

#### Acceptance Criteria
- DuckDB boundary test passes (no non-storage DuckDB usage).
- SCIP tooling tests pass without requiring a git repo.
- `build.status` and other read operations meet performance budget.
- Subprocess registry tests pass and semantics are clear.

#### Likely Touchpoints
- `src/codeintel/build/serving/publisher.py`
- `src/codeintel/storage/serving/`
- `src/codeintel/ingestion/engine/scip.py`
- `src/codeintel/observability/runtime_registry.py`
- CLI performance harness

#### Phase 5 Tests
- `tests/architecture/test_duckdb_boundaries.py`
- `tests/ingestion/test_tools.py`
- `tests/ingestion/test_tool_runner_registry.py`
- `tests/cli/performance/test_performance.py`

## 7) Cross-Cutting Test Hygiene
- Remove forbidden monkeypatch usage in `tests/cli/test_cli_telemetry.py`.
- Prefer test harness injection over patching.

## 8) Validation Plan (After Each Phase)
- Phase 1: CLI + storage gateway validation tests, contract validation tests.
- Phase 2: dataset contract snapshot tests + manifest tests.
- Phase 3: CLI help/flags/snapshots.
- Phase 4: analytics + goldens.
- Phase 5: boundaries + tooling + performance.

## 9) Risks and Mitigations
- Risk: Lenient validation masks real errors.
  - Mitigation: enforce strict validation for explicit commands and CI checks.
- Risk: CLI flag redesign breaks snapshots again.
  - Mitigation: add a dedicated env var stability test and update snapshots once.
- Risk: Schema alignment changes alter analytics outputs.
  - Mitigation: update goldens and document changes as intentional.
- Risk: Tooling changes require external binaries.
  - Mitigation: use stubs and fallbacks in tests.

## 10) Deliverables
- Updated storage validation strategy with explicit validation mode.
- DAG artifact for schema inference errors.
- Stable shared CLI flags and explicit env vars.
- Schema-aligned analytics writers and updated goldens.
- Storage-owned serving snapshot service and robust tooling behaviors.
