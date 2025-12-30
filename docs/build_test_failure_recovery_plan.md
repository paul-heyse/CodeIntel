# Build Test Failure Recovery Plan (Goids + Profiles as Hamilton Targets)

## Context
- `goids` and `profiles` are **Hamilton targets**, derived from data inputs (not external inputs).
- Current `tests/build` failures stem from a mix of cache invalidation, planner input gaps, missing target modules, and stale test expectations.
- This plan focuses on fixing the root causes while aligning with the dynamic schema inference + Iceberg-first materialization path.

## Objectives
- Restore `goids` and `profiles` as first-class Hamilton targets with stable outputs/tags.
- Eliminate systemic failures in planning, schema inference, materialization, and tag validation.
- Update test suites to reflect current module structure and node naming rules.
- Regain a clean, deterministic `tests/build` run with the smallest set of evergreen invariants.

## Non-Goals
- Large-scale refactors unrelated to the failing tests.
- Reintroducing legacy orchestrators or deprecated module structures.
- Broad documentation updates outside this plan.

## Execution Plan

### Phase 0: Baseline + Guardrails
- Capture current test state and environment:
  - `uv run pytest tests/build -q` (retain `build/test-results/*`).
  - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- Confirm existing snapshot expectations and determine whether snapshots need update or output normalization.

### Phase 1: Stabilize Contract/Schema Service Lifecycles
**Problem**: `clear_contract_cache()` and `clear_target_metadata_cache()` fully reset global services, breaking tests expecting a configured service.

**Changes**
- Update `src/codeintel/build/schemas/contract_service.py`:
  - Make `clear_contract_cache()` clear LRU caches only.
  - Add a new explicit reset helper (e.g., `reset_contract_service()` or `clear_contract_service_state()`).
- Update `src/codeintel/build/target_metadata.py`:
  - Make `clear_target_metadata_cache()` clear provider caches without removing provider registrations.
  - Add a new explicit reset helper for tests requiring a hard reset.
- Update `src/codeintel/build/schemas/row_registry.py` to avoid raising when schema service is unconfigured; only clear caches if configured.

**Tests**
- Update `tests/build/hamilton/test_pr68_contract_provider_parity.py`,
  `tests/build/test_contract_resolution_seams.py`,
  `tests/build/hamilton/test_import_time_schema_safety.py`
  to call the explicit reset helper when they truly need a fresh service.

### Phase 2: Fix Planner Inputs for None Values
**Problem**: `compute_plan()` drops `cache_index` / `cache_key_resolver` when they are `None`, but planning nodes declare them as required.

**Changes**
- Update `src/codeintel/build/hamilton/planner.py`:
  - Always include `cache_index` and `cache_key_resolver` in the input mapping (even when `None`).
  - Preserve current behavior for other optional inputs.
- (Optional alternative) Add explicit defaults in `src/codeintel/build/hamilton/native/planning/plan_nodes.py`.

### Phase 3: Schema Inference + Materialization Compatibility
**Problem**: materializers accept `DuckDBRelation` and `pa.Table`, but schema helpers only handle `LazyFrame`/`RecordBatchReader`. Also inference should accept `pl.DataFrame` directly or coerce to `LazyFrame`.

**Changes**
- Update `src/codeintel/build/hamilton/materializers/columnar_utils.py`:
  - Extend `table_schema_for_data()` to accept `pa.Table` and `DuckDBRelation` by converting to `RecordBatchReader`.
  - Extend `arrow_schema_for_data()` to accept `pa.Table` and `DuckDBRelation`.
- Update `src/codeintel/build/schemas/inference_service.py`:
  - Accept `pl.DataFrame` for `table_schema_from_tabular()` or coerce `pl.DataFrame` → `LazyFrame`.
- Keep inference strict for truly unsupported types.

**Tests**
- Update `tests/build/schemas/test_inference_service_arrow_polars.py`.
- Validate `tests/build/hamilton/test_materializer.py::test_iceberg_saver_accepts_duckdb_relation`.

### Phase 4: Reintroduce `goids` Target (Core Dataset)
**Goal**: Provide a native Hamilton target that outputs `core.goids` with correct tags and schema.

**Changes**
- Add a native Hamilton module (recommended: `src/codeintel/build/hamilton/native/ingestion/goids_targets.py`
  or `.../graphs/goids_targets.py`) that:
  - Consumes `core.ast_nodes` / `core.modules` (and any required tool outputs).
  - Uses existing GOID computation functions (e.g., `codeintel/graphs/compute/goid.py` or analytics GOID loaders).
  - Produces a single `core.goids` dataset via `save_dataset`.
  - Exposes `t__goids` target with `node_type=materialize` and proper domain/target tags.
- Update target catalog output inventory if needed (e.g., `src/codeintel/core/registry/dag_output_inventory.yaml`).
- Ensure `q__core__goids` is auto-generated and tagged with `node_type=loader.query`.

**Tests**
- Update/verify `tests/build/test_registry.py`, `tests/build/test_hamilton_phase0.py`.
- Ensure `tests/build/hamilton/test_pr64_loader_tags_are_canonical.py` passes.

### Phase 5: Reintroduce `profiles` Target (Analytics Datasets)
**Goal**: Restore a first-class `profiles` Hamilton target to materialize analytics profile tables.

**Changes**
- Add a native Hamilton module (recommended: `src/codeintel/build/hamilton/native/analytics/tables_profiles.py`) that:
  - Produces `analytics.function_profile`, `analytics.file_profile`, `analytics.module_profile`, `analytics.test_profile`.
  - Uses existing analytics builders in `codeintel/analytics/testing/profiles/*` and related compute modules.
  - Persists datasets via `save_dataset`, with `target=profiles` and canonical tags.
- Ensure all dataset outputs are registered in the catalog.
- Confirm dependencies are consistent with `function_metrics`, `risk_factors`, `coverage_functions`, and testing tables.

**Tests**
- Update/verify `tests/build/test_registry.py` (profiles dependency expectations).
- Update target-name expectations where necessary.

### Phase 6: Tagging + Support Nodes Consistency
**Problem**: Loader and dataset nodes lack canonical `node_type` tags; support nodes can ignore config flags.

**Changes**
- Update `src/codeintel/build/hamilton/nodes/support_nodes.py`:
  - Ensure `ci_support_include_loader_nodes=False` truly suppresses q__ nodes.
  - Validate tag propagation for loader nodes (`node_type=loader.query`, `table_key`).
- Update tagging for ingestion table nodes:
  - Ensure `ast__node_rows`, `ast__metric_rows`, `cst__node_rows`, `docstrings__rows` are tagged as datasets or update tests to validate saver outputs instead of raw nodes.

**Tests**
- Update `tests/build/hamilton/test_pr64_all_nodes_have_node_type_tag.py` to ignore Hamilton macro-generated nodes with non-identifier names (e.g., `.raw`), or normalize those names in generation.

### Phase 7: Remove `env.gateway` From Compute Nodes
**Problem**: Analytics compute nodes access `env.gateway` directly, violating the compute-node boundary.

**Changes**
- Refactor:
  - `src/codeintel/build/hamilton/native/analytics/tables_coverage.py`
  - `src/codeintel/build/hamilton/native/analytics/tables_dependencies.py`
  - `src/codeintel/build/hamilton/native/analytics/tables_functions.py` (if needed)
  to accept `TabularInput` dependencies via `q__*` loader nodes and convert via `to_lazyframe`.
- Keep all gateway access inside loader nodes or materializers.

**Tests**
- `tests/build/hamilton/test_no_env_gateway_in_compute_nodes.py`.

### Phase 8: Test Infrastructure + Dependency Fixes
- Register `ModuleType` modules in `sys.modules` in:
  - `tests/build/hamilton/test_dag_catalog_compiler.py`
  - `tests/build/hamilton/test_saver_declared_output_inventory.py`
- Update Polars equality in `tests/build/hamilton/test_tabular_steps.py` to use supported API.
- Update `_FakeInferenceService` signature to match `SchemaInferenceService`.
- Add `duckdb-engine` dependency for SQLAlchemy dialect (or conditionally skip tests):
  - `tests/build/hamilton/test_materializer.py` (SQLAlchemy-based checks).

### Phase 9: Snapshot + Taxonomy Alignment
- Regenerate CLI snapshots or normalize output format.
- Update snapshot manifest tags:
  - Replace invalid tag `targets` with a valid taxonomy tag (e.g., `list`, `status`, or `plan`).
  - Update `tests/build/hamilton/test_pr55_final_sweep.py` expectations accordingly.

## Acceptance Criteria
- `tests/build` passes (or only skips for explicitly missing optional deps).
- `goids` and `profiles` appear as Hamilton targets with correct tags and output inventory.
- `q__core__goids` exists and is tagged as `loader.query`.
- Materializers accept `DuckDBRelation` and `pa.Table` for schema inference.
- No compute nodes access `env.gateway`.
- CLI snapshot taxonomy and output are consistent.

## Validation Steps
1. `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
2. `uv run pytest tests/build/hamilton -q`
3. `uv run pytest tests/build/schemas -q`
4. `uv run pytest tests/build/serving -q`
5. `uv run pytest tests/build -q` (full build suite)

## Risk Notes
- Reintroducing `goids`/`profiles` requires careful dependency ordering to avoid cyclic target dependencies.
- Updating node tagging will cascade into catalog outputs; keep tag validation strict and aligned with `TagSpec`.
- Snapshot updates should only be done after functional fixes to avoid masking regressions.
