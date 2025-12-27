# Systemic Pytest Failures Remediation Plan (v2)

## Purpose
This plan addresses the remaining pytest failures by fixing systemic root causes and
aligning the runtime with the DAG-first design. It incorporates the clarified
decisions:

- Graph targets missing optional inputs should be **missing/skipped**, not silently
  succeeded.
- JSON columns should be stored as **native JSON in DuckDB**, with JSON encoding
  only at export boundaries.

## Objectives
- Eliminate remaining failures by correcting core data flow contracts, row handling,
  and tooling assumptions.
- Improve reliability and maintainability via clear boundary responsibilities.
- Preserve DAG-first behavior and deterministic build state tracking.

## Scope Map (Failure Clusters -> Root Causes)
1) **Row serialization / JSON column handling**
   - Ingestion serializer test expects JSON arrays, but JSON is being stringified.
   - Export validation fails (function_profile, coverage_functions, function_validation).

2) **core.modules row_hash requirements**
   - Storage rejects rows missing `row_hash` for core.modules.
   - Impacts module index and graph prerequisites.

3) **Graph/docs view targets failing with missing inputs**
   - Call graph fails due to missing `core.file_state` or upstream ingestion.
   - Cascades to docs views and export edge columns tests.

4) **SCIP tool resolution**
   - Tool resolution uses `scip` binary name; `scip-python` is canonical.
   - `scip_proto__hash_options` failing indicates tooling mismatch or missing inputs.

5) **CLI contract drift**
   - `computed` block missing from build status JSON.
   - Resolution missing-params not raising errors.

6) **Observability config/metrics**
   - config_file not preserved, config validation too strict, metrics registry not
     emitting expected instruments, attribute budget overrun.

## Plan Overview (Priority Order)
1) Row serialization & JSON boundary split + row_hash policy.
2) Graph/input prerequisite handling (missing inputs -> missing/skipped).
3) SCIP tool unification and resilient hash options.
4) CLI contract stabilization (status JSON + strict resolution).
5) Observability config/metrics robustness (align with test expectations).

## Phase 1: Row Serialization & Row Hash Policy (Highest Priority)

### Goal
Store JSON as native objects in DuckDB, encode only on export, and ensure
row_hash is always present when required.

### Design Changes
1) **Split row serialization from storage encoding**
   - Row serializer should **not** encode JSON values; it should only normalize
     missing/NaN values and basic numpy/pandas conversions.
   - JSON encoding should occur only in export writers or explicit JSONL adapters.

2) **Centralize row_hash policy**
   - If schema includes `row_hash` and value missing, compute it from canonical
     row mapping (excluding row_hash) in the write path.
   - Avoid duplicating manual row_hash logic in individual ingestion steps.

### Implementation Tasks
- **Row serializer adjustments**
  - Update `src/codeintel/core/schemas/row_models.py` so
    `normalize_row_value_for_type(..., JSON)` returns native objects, not JSON
    strings.
  - Ensure `RowSerializer` path is schema-order only.
  - Add a distinct export encoder for JSON (see Phase 1.3).

- **Row hash policy**
  - Add a write-path hook (policy backend or insert helpers) that injects
    `row_hash` when missing and required by schema.
  - Ensure hash uses a deterministic, ordered mapping (schema order).

- **Export encoding**
  - Update export writers in `src/codeintel/build/exports/*` or
    `src/codeintel/storage/io/*` to JSON-encode values for JSON columns only.
  - Keep DB storage native (dict/list), JSONL output stringified.

### Tests / Acceptance
- `tests/ingestion/test_row_serialization.py::test_ingestion_row_serializer_matches_schema_order`
  should pass with JSON lists (not strings).
- Export validation tests should pass:
  - `tests/docs_export/test_export_smoke.py::test_export_validation_passes_on_minimal_data`
  - `tests/docs_export/test_function_validation_export.py::test_function_validation_export`
  - `tests/docs_export/test_graph_validation_export.py::test_graph_validation_export`
- `tests/storage/test_module_index.py::test_load_module_map_filters_by_language`
  should pass (row_hash present).

## Phase 2: Graph Inputs & Missing/Skipped Semantics

### Goal
Missing optional inputs should mark targets as missing/skipped, not silently
successful, and should be clearly reflected in TargetRunRecord status.

### Design Changes
1) **Explicit input classification**
   - Identify required vs optional inputs for graph targets.
   - Missing required inputs => target status `missing` (or `skipped` with reason).
   - Missing optional inputs => `skipped` with clear reason.

2) **Preflight status propagation**
   - Preflight should mark blocked graph targets based on missing inputs.
   - Ensure downstream targets inherit missing/skipped state.

### Implementation Tasks
- Update preflight in `src/codeintel/build/hamilton/executor.py` to:
  - Distinguish optional vs required input keys.
  - Emit `TargetRunRecord` with status `skipped`/`missing_input`.
- Update graph targets in:
  - `src/codeintel/build/hamilton/native/graphs/call_graph.py`
  - `src/codeintel/build/hamilton/native/graphs/import_graph.py`
  - `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
  - to declare optional inputs explicitly (e.g., file_state).
- Update docs view orchestration to respect missing/skipped graph targets.

### Tests / Acceptance
- Graph-related tests should pass or correctly show `missing/skipped`:
  - `tests/graphs/test_callgraph_builder.py`
  - `tests/graphs/test_engine_nx.py`
  - `tests/graphs/test_span_consistency_integration.py`
- Docs view tests should pass:
  - `tests/storage/test_docs_views.py::*`
  - `tests/storage/test_docs_view_profiling.py::*`

## Phase 3: SCIP Tool Unification

### Goal
Canonicalize tool resolution to `scip-python` and avoid failures from missing
binary names.

### Design Changes
1) **Tool registry aliasing**
   - Treat `scip` and `scip-python` as the same tool; resolve to scip-python.

2) **Resilient hash options**
   - Allow `scip_proto__hash_options` to compute without strict file_state
     coupling when not available.

### Implementation Tasks
- Update tool config resolution to register `scip-python` as canonical name and
  accept `scip` as alias.
- Adjust ingestion nodes in `src/codeintel/build/hamilton/native/ingestion/scip.py`
  to compute hash options without requiring full file_state context.

### Tests / Acceptance
- `tests/ingestion/test_scip_ingest.py::test_scip_target_writes_tables` passes.
- No more tool-missing skips for scip-related targets.

## Phase 4: CLI Contract Stabilization

### Goal
Restore stable CLI payloads and strict resolution errors.

### Design Changes
1) **Status JSON includes computed block**
   - Reintroduce `computed` field containing computed targets/summary.

2) **Resolution validation strictness**
   - Missing required params should raise `ResolutionError`.

### Implementation Tasks
- Update `src/codeintel/cli/handlers/build.py` to include `computed` in
  status JSON output.
- Update resolution logic in `src/codeintel/cli/resolution/*` and
  `src/codeintel/cli/services/*` to enforce required params.

### Tests / Acceptance
- `tests/cli/test_build_cli.py::TestBuildStatusCommand::test_status_json_output_structure`
- `tests/cli/unit/test_resolution_integration.py::test_resolution_missing_params_raises_error`

## Phase 5: Observability Config + Metrics Robustness

### Goal
Allow minimal config files, preserve config_file path, ensure metrics registry
emits expected instruments, and enforce budget truncation correctly.

### Design Changes
1) **Config validation**
   - Accept minimal or empty config files when observability is enabled in tests.
   - Preserve `config_file` path on resolved settings.

2) **Metrics registration**
   - Ensure metrics registry initializes instruments even in test mode.

3) **Attribute budget enforcement**
   - Enforce deterministic truncation of attributes above budget.

### Implementation Tasks
- Update config parsing/validation in `src/codeintel/observability/*`.
- Ensure `ObservabilitySettings.config_file` is populated after resolution.
- Update metrics view registration for test/in-memory setups.
- Adjust attribute normalizer budget logic to truncate at limits.

### Tests / Acceptance
- `tests/observability/test_otel_config.py::*`
- `tests/observability/test_metrics_views.py::test_build_views_emits_expected_instruments`
- `tests/observability/test_attribute_normalizer.py::test_normalizer_enforces_budget_limits`
- `tests/observability/test_config_resolver.py::test_config_resolver_snapshot_matches_golden`

## Test Strategy (By Phase)
1) Phase 1:
   - `tests/ingestion/test_row_serialization.py`
   - `tests/storage/test_module_index.py`
   - `tests/docs_export/test_export_smoke.py`
2) Phase 2:
   - `tests/graphs/test_callgraph_builder.py`
   - `tests/graphs/test_engine_nx.py`
   - `tests/storage/test_docs_views.py`
3) Phase 3:
   - `tests/ingestion/test_scip_ingest.py`
4) Phase 4:
   - `tests/cli/test_build_cli.py`
   - `tests/cli/unit/test_resolution_integration.py`
5) Phase 5:
   - `tests/observability/test_otel_config.py`
   - `tests/observability/test_metrics_views.py`
   - `tests/observability/test_attribute_normalizer.py`

## Risks & Mitigations
- **Risk:** JSON encoding changes may break existing DB expectations.
  - **Mitigation:** encode only at export boundaries; keep DB storage native.
- **Risk:** Missing input classification could mask real failures.
  - **Mitigation:** explicit required vs optional input registry and clear
    `TargetRunRecord` status reason strings.
- **Risk:** Tool resolution changes may affect other tooling.
  - **Mitigation:** alias support + fallback to explicit tool version checks.

## Acceptance Checklist
- All failures in the latest pytest summary are resolved.
- JSON columns are native in DuckDB; exports encode JSON correctly.
- Graph targets with missing inputs are marked missing/skipped with reason.
- SCIP tool resolution uses `scip-python` canonical name.
- CLI status JSON includes `computed`; missing params raise `ResolutionError`.
- Observability config tests pass with minimal config files.

## Notes
- This plan assumes continued DAG-first alignment: schemas and row bindings must
  derive from the Hamilton DAG output contracts whenever available.
- If any phase reveals additional upstream constraints, add a short addendum
  section with the updated dependency and test slice.
