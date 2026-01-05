---
title: Systemic Pytest Failures Implementation Plan
status: draft
owner: codeintel
last_updated: 2025-12-26
---

# Systemic Pytest Failures Implementation Plan

## Goals

- Resolve the clustered pytest failures with a small set of high-leverage fixes.
- Harden schema authority and contract alignment to be DAG-first across build, storage, and serving.
- Improve maintainability and extensibility through clear sources of truth and guardrails.

## Non-Goals

- Rewriting observability from scratch.
- Backward compatibility with deprecated schema paths (design phase allows aggressive migration).

## Guiding Principles

- **Single schema authority**: Hamilton-derived schema wins when present.
- **Contract catalog = published schema**: contract catalog should always reflect SchemaService.
- **Target failure transparency**: preflight failures should be explicit and actionable.
- **Schema alignment checks are first-class guardrails**, not ad hoc debug steps.
- **Stable CLI semantics**: table_key is the canonical identifier in CLI output.

## Current Failure Clusters (Root-Cause Summary)

1. Missing `impl_kind` column in tracking tables → binder errors in CLI/status and multiple suites.
2. SchemaService vs contract catalog mismatch (notably `core.modules`) → contract validation failures.
3. Target execution failures cascading into graph/docs/export tests → upstream input gaps and validation errors.
4. Docs views registry gaps (e.g., `schema_inference_errors`) → registry validation failures.
5. CLI dataset identifier drift (name vs table_key) → dataset info/flow failures.
6. JSON columns serialized as Python strings → malformed JSON errors.
7. Observability runtime disabled or miswired → no spans/log handlers/metrics.
8. Insert helper leniency regression → missing column checks not enforced.

## Implementation Sequence Overview

1. Schema authority and catalog regeneration
2. Tracking schema migration (`impl_kind`)
3. Build preflight and failure semantics
4. Docs view/registry cohesion
5. CLI dataset identifier normalization
6. JSON column serialization contract
7. Observability runtime hardening
8. Write-path strictness and validation

Each phase includes explicit acceptance gates and test slices.

---

## Phase 1: Schema Authority + Contract Catalog Regeneration

### Objective
Eliminate SchemaService vs contract catalog mismatch and establish DAG-first schema authority.

### Work Items

- **Schema provenance and refresh pipeline**
  - Ensure the contract catalog is rebuilt from SchemaService output when inference is enabled.
  - Add provenance metadata (declared vs inferred) to table schemas to control strict equality.
- **Alignment guardrail**
  - Keep the SchemaService vs contract catalog check for non-view tables.
  - Fail fast if any mismatch remains (no silent fallback).
- **Publish path**
  - Update the catalog publish step to use SchemaService (not raw config schemas).

### Candidate Files

- `src/codeintel/build/schemas/service.py`
- `src/codeintel/storage/contracts/provider.py`
- `src/codeintel/storage/contracts/catalog_state.py`
- `src/codeintel/core/schemas/service.py`
- `src/codeintel/core/schemas/contract_validation.py`

### Acceptance Criteria

- No `SchemaService schema mismatch` errors in export or conformance tests.
- Contract catalog, SchemaService, and DDL schemas are identical for non-view tables.

### Tests

- `tests/config/test_dataset_contract.py::test_schema_service_matches_contract_catalog`
- `tests/storage/test_conformance.py::test_conformance_passes_with_empty_db`

---

## Phase 2: Tracking Schema Migration (`impl_kind`)

### Objective
Add `impl_kind` to tracking tables and make tracking queries resilient to schema drift.

### Work Items

- **DDL and migration**
  - Add `impl_kind` column to tracking tables and metadata DDL.
  - Update `tools/migrate_impl_kind_columns.py` (or add a new migration) to backfill.
- **Query resilience**
  - Use schema-aware select lists or compatibility view when column is missing.
  - Ensure CLI/build status queries don’t fail on old DBs.

### Candidate Files

- `src/codeintel/core/schemas/table_registry.py`
- `src/codeintel/storage/tracking/build_tracking.py`
- `src/codeintel/storage/metadata/ddl.py`
- `tools/migrate_impl_kind_columns.py`

### Acceptance Criteria

- No `_duckdb.BinderException: impl_kind not found` failures.
- Build status CLI works on new and migrated DBs.

### Tests

- `tests/cli/test_build_cli.py::TestBuildStatusCommand::test_status_json_output_structure`
- `tests/cli/golden/test_golden_output.py::test_build_status_text_output`

---

## Phase 3: Build Preflight + Target Failure Semantics

### Objective
Make target failures explicit (missing inputs vs execution errors) and prevent cascading failures.

### Work Items

- **Preflight check**
  - Validate required upstream tables exist before running targets.
  - Example: call graph depends on file_state and module inventory.
- **Failure classification**
  - Distinguish “missing input” from “execution error” in TargetRunRecord.
  - Keep failure reason visible in CLI status.

### Candidate Files

- `src/codeintel/build/hamilton/executor.py`
- `src/codeintel/build/hamilton/run_records.py`
- `src/codeintel/build/hamilton/planner.py`
- `src/codeintel/storage/tracking/build_tracking.py`

### Acceptance Criteria

- Target status “failed” includes explicit missing input reason.
- Graph/doc/export tests stop failing due to unclear “target_status failed”.

### Tests

- `tests/graphs/test_callgraph_builder.py::test_callgraph_handles_aliases_and_relative_imports`
- `tests/storage/test_docs_views.py::test_docs_views_registered_in_metadata`

---

## Phase 4: Docs Views and Registry Cohesion

### Objective
Ensure docs views are registered, created, and validated consistently.

### Work Items

- **Registry alignment**
  - Ensure all docs views in contract catalog are registered in metadata.
  - Include `schema_inference_errors` and any inferred views.
- **Bootstrap behavior**
  - Ensure docs views are created during metadata bootstrap when requested.

### Candidate Files

- `src/codeintel/storage/views/inventory.py`
- `src/codeintel/storage/metadata/bootstrap.py`
- `src/codeintel/storage/datasets/registry.py`

### Acceptance Criteria

- No missing view errors during export/validation.
- Docs view tests pass without manual seeding.

### Tests

- `tests/storage/test_docs_views.py`
- `tests/docs_export/test_export_smoke.py::test_export_validation_passes_on_minimal_data`

---

## Phase 5: CLI Dataset Identifier Normalization

### Objective
Make CLI outputs consistently use `table_key` (schema.table) as the canonical ID.

### Work Items

- **Dataset reference model**
  - Introduce a `DatasetRef` or normalize responses to table_key.
  - Resolve input by name, but output table_key.
- **Handler alignment**
  - Dataset info/flow handlers should emit table_key and schema info together.

### Candidate Files

- `src/codeintel/cli/handlers/datasets.py`
- `src/codeintel/cli/commands/*`
- `src/codeintel/cli/core/result_types.py`

### Acceptance Criteria

- CLI dataset info/flow tests pass with table_key outputs.

### Tests

- `tests/cli/commands/test_dataset_info_flow.py`
- `tests/cli/handlers/test_datasets.py`

---

## Phase 6: JSON Column Serialization Contract

### Objective
Ensure JSON columns are always serialized as valid JSON strings.

### Work Items

- **Centralized serializer**
  - Route JSON column serialization through row bindings or a shared serializer.
  - Convert lists/dicts to JSON strings before insertion.
- **Validation**
  - Add enforcement that JSON columns are not raw Python stringified lists.

### Candidate Files

- `src/codeintel/core/schemas/row_models.py`
- `src/codeintel/storage/warehouse.py`
- `src/codeintel/storage/io/ibis_io.py`

### Acceptance Criteria

- No JSON conversion errors when writing analytics profiles or tags.

### Tests

- `tests/analytics/test_profiles_and_functions.py::test_test_and_behavioral_profile_writers`

---

## Phase 7: Observability Runtime Hardening

### Objective
Ensure spans, logs, and metrics are emitted when enabled, and a stable no-op runtime exists when disabled.

### Work Items

- **Runtime interface compliance**
  - Provide a tracer provider with required SDK surface (`add_span_processor`).
  - Ensure log handler and meter provider are attached when enabled.
- **Config precedence**
  - Explicitly honor config file overrides for SDK enablement.

### Candidate Files

- `src/codeintel/observability/otel.py`
- `src/codeintel/observability/runtime.py`
- `src/codeintel/observability/logs.py`

### Acceptance Criteria

- Observability smoke tests record spans and attach log handlers.

### Tests

- `tests/observability/test_observability_smoke.py`
- `tests/observability/test_logs_pipeline.py`
- `tests/observability/test_otel_config.py`

---

## Phase 8: Write-Path Strictness and Validation

### Objective
Restore strict enforcement of required columns during inserts, independent of lenient contract validation.

### Work Items

- **Insert validation**
  - Re-enforce missing column checks on insert helpers.
  - Keep lenient validation in registry checks only.

### Candidate Files

- `src/codeintel/storage/warehouse.py`
- `src/codeintel/storage/gateway/base_accessor.py`
- `src/codeintel/storage/io/ibis_io.py`

### Acceptance Criteria

- Insert helper tests raise on missing columns.

### Tests

- `tests/storage/test_insert_helpers.py::test_insert_rows_raises_on_missing_column`

---

## Cross-Cutting Improvements

- **Contract catalog provenance**: annotate schemas with `source: declared|inferred`.
- **DAG-first enforcement**: anywhere a static schema competes with inferred schema, prefer inferred if non-empty.
- **Compatibility views**: for tracking tables, use views to bridge schema migrations.
- **Diagnostics artifact**: create a short-lived diagnostics payload during validation failures for postmortems.

## Risks and Mitigations

- **Schema mismatches across environments**: mitigate with explicit catalog regeneration and strict guards.
- **Migration sequencing**: ensure `impl_kind` migration lands before build status queries.
- **Observability changes overlap**: coordinate with ongoing telemetry changes.

## Success Criteria

- Eliminate all current clustered failures in pytest run.
- Contracts and SchemaService are aligned across all non-view tables.
- CLI and docs/export flows work without schema drift.

