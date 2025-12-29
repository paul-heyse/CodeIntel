# Best-in-Class Apache Hamilton Implementation Plan

## Goal

Deliver a best-in-class Hamilton implementation that maximizes DAG observability,
schema and validation rigor, configurable execution, and UI/catalog fidelity while
keeping current business logic stable and evolvable.

## Scope Summary

- Complete schema and tag metadata coverage for table and artifact outputs.
- Output validation using Hamilton data-quality modifiers with warn vs fail modes.
- Materialization tags aligned with UI catalog expectations.
- Tag-driven discovery and execution in the CLI and runtime.
- Cross-cutting normalization using mutate and pipe patterns.
- Reusable pipelines via parameterize/parameterized_subdag and resolve_from_config.
- Optional dynamic execution with Parallelizable/Collect.
- Expanded adapter support (threadpool plus optional Ray/Dask) and ResultBuilder
  for large outputs.
- Builder-level materializer injection for environment-specific I/O.
- Cache lineage export and audit ingestion into DuckDB.
- Hamilton UI telemetry capture policy hardening.

## Non-Goals

- Rewriting the full DAG or changing semantic data models.
- Async Hamilton nodes (explicitly disallowed by current validation).
- OpenLineage integration (out of scope per guidance).
- Forcing any production rollout without explicit enablement flags.

## Assumptions and Dependencies

- `hamilton.enable_power_user_mode` remains enabled in `src/codeintel/runtime/compose.py`.
- Hamilton SDK is optional; tracker wiring must remain resilient to missing extras.
- The tagging system in `src/codeintel/build/hamilton/tagging.py` remains canonical.
- Cache log ingestion is available via `src/codeintel/observability/cache_log_ingest.py`.

## Implementation Phases

### Phase 0 - Inventory and Design Mapping

Tasks
- Inventory all Hamilton target modules and outputs (tables, views, artifacts).
- Map each output to a schema contract source (declared or inferred).
- Identify repeated pipeline patterns suitable for parameterized_subdag.

Primary files
- `src/codeintel/build/hamilton/native/**`
- `src/codeintel/build/hamilton/materializers/**`
- `src/codeintel/build/schemas/observations.py`
- `src/codeintel/serving/semantic/registry_compiler.py`

Acceptance
- A tracked mapping of outputs to tags, schema hints, and materialization type.

### Phase 1 - Schema and Tag Metadata Foundation

Tasks
- Apply `schema_output(...)` on all table-producing nodes via
  `src/codeintel/sdk/annotations.py`.
- Enforce tag completeness for table outputs (domain, target, table_key,
  output_kind).
- Add materialization-related tags on saver nodes where missing.

Primary files
- `src/codeintel/sdk/annotations.py`
- `src/codeintel/build/hamilton/native/ingestion/**`
- `src/codeintel/build/hamilton/native/graphs/**`
- `src/codeintel/build/hamilton/native/analytics/**`
- `src/codeintel/build/hamilton/native/export/**`
- `src/codeintel/build/hamilton/tagging.py`
- `src/codeintel/build/hamilton/tag_spec.py`

Acceptance
- All table outputs expose schema hints via `hamilton.internal.schema_output`.
- Tag validation passes for every table materializer node.

### Phase 2 - Output Validation with Data Quality Modifiers

Tasks
- Apply `check_output_warn` and `check_output_fail` for key invariants
  (null count, row count, primary keys) using `src/codeintel/sdk/validation.py`.
- Gate strict validation behavior via Hamilton config keys and CLI flags.
- Tag validation nodes so they can be hidden in UI visualizations.

Primary files
- `src/codeintel/sdk/validation.py`
- `src/codeintel/build/hamilton/validate.py`
- `src/codeintel/build/hamilton/native/**`
- `src/codeintel/build/hamilton/tagging.py`

Acceptance
- Validation failures are surfaced with explicit node metadata and error codes.
- Validation mode can be switched between warn and fail without code changes.

### Phase 3 - Materialization Tags and Catalog Alignment

Tasks
- Emit `materialization` and `materialized_name` tags for saver nodes.
- Ensure materializer outputs align with UI catalog expectations.
- Wire materialization tags into semantic registry compilation if needed.

Primary files
- `src/codeintel/build/hamilton/save_to.py`
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
- `src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py`
- `src/codeintel/build/hamilton/materializers/artifact_saver.py`
- `src/codeintel/serving/semantic/registry_compiler.py`

Acceptance
- UI catalog can map `semantic_id` to materialized object name and type.
- Materialized nodes are consistently discoverable by tag query.

### Phase 4 - Tag-Driven CLI and Runtime Querying

Tasks
- Add `--tag` filtering for target listing and build execution.
- Use `TagQuery` to resolve tag-driven selections deterministically.
- Provide `--show-origin` and tag metadata in CLI outputs.

Primary files
- `src/codeintel/cli/commands/build.py`
- `src/codeintel/cli/commands/targets.py`
- `src/codeintel/cli/handlers/build.py`
- `src/codeintel/core/hamilton/tag_query.py`

Acceptance
- CLI can run or list targets by tag without manual name enumerations.
- Tag filters are cached and consistent across runs.

### Phase 5 - Cross-Cutting Normalization with mutate

Tasks
- Introduce mutate-based normalization for ingestion pipelines
  (eg. drop nulls, normalize columns, add defaults).
- Centralize mutation logic in a dedicated transforms module and
  apply via `pipe_input` and `mutate` decorators.

Primary files
- `src/codeintel/build/hamilton/native/ingestion/pipelines.py`
- `src/codeintel/build/hamilton/transforms/**`

Acceptance
- Ingestion cleanup is declarative, reusable, and tagged for lineage.

### Phase 6 - Parameterized SubDAGs and Config-Driven Composition

Tasks
- Replace repeated pipeline patterns with `parameterized_subdag` or
  `parameterize_sources`.
- Use `resolve_from_config` to select feature sets at compile time.
- Keep config keys namespaced and validated.

Primary files
- `src/codeintel/build/hamilton/native/ingestion/**`
- `src/codeintel/build/hamilton/native/graphs/**`
- `src/codeintel/build/hamilton/nodes/support_nodes.py`
- `src/codeintel/runtime/compose.py`

Acceptance
- Pipeline variants are selected by config without code duplication.
- SubDAG creation is deterministic and validated.

### Phase 7 - Dynamic Execution with Parallelizable/Collect

Tasks
- Identify a safe Parallelizable/Collect use case (eg. per-module ingest).
- Implement dynamic execution path guarded by config.
- Provide explicit local and remote executor configuration in runtime config.

Primary files
- `src/codeintel/runtime/compose.py`
- `src/codeintel/build/hamilton/executor.py`
- `src/codeintel/build/hamilton/native/ingestion/**`

Acceptance
- Dynamic execution is off by default and fully gated by config.
- When enabled, results are correct and parallel execution is observable.

### Phase 8 - Adapter Expansion and ResultBuilder

Tasks
- Add optional Ray/Dask adapters to `create_parallel_adapter` with safe detection.
- Introduce a ResultBuilder for large outputs to avoid oversized dict returns.
- Add adapter-level safeguards for materialize nodes (lock or task grouping).

Primary files
- `src/codeintel/build/hamilton/adapters/parallel.py`
- `src/codeintel/build/hamilton/executor.py`
- `src/codeintel/build/hamilton/driver_options.py`

Acceptance
- Optional adapters are discoverable and do not error when deps are missing.
- ResultBuilder can be selected without changing target code.

### Phase 9 - Builder-Level Materializers

Tasks
- Extend `BuildDriverOptions` to accept materializer specs.
- Wire `Builder.with_materializers(...)` in `compose_runtime`.
- Add config schema for materializer selection per environment.

Primary files
- `src/codeintel/build/hamilton/driver_options.py`
- `src/codeintel/runtime/compose.py`
- `src/codeintel/core/runtime/loader.py`

Acceptance
- Materializers can be attached via config only, no code edits required.
- Materialization nodes appear in DAG and are taggable.

### Phase 10 - Cache Lineage Export and Audit Pipelines

Tasks
- Export cache lineage to DuckDB after build runs using cache logs and metadata.
- Link cache events to run IDs and target outputs for auditability.
- Provide a CLI hook for manual cache log ingestion.

Primary files
- `src/codeintel/build/hamilton/executor.py`
- `src/codeintel/observability/cache_log_ingest.py`
- `src/codeintel/cli/handlers/storage.py`
- `src/codeintel/cli/commands/storage.py`

Acceptance
- Cache lineage is queryable in DuckDB by run_id and node_name.
- Cache log ingestion succeeds for both cache_dir and explicit jsonl paths.

## Remaining Work Checklist (Ordered)

### Phase 0 - Inventory and Design Mapping
- [ ] Build an output-to-schema inventory artifact (tables, views, artifacts) capturing schema source-of-truth and tags. Files: `docs/hamilton_best_in_class_inventory.md`, `src/codeintel/build/hamilton/native/**`, `src/codeintel/build/schemas/**`
- [ ] Catalog repeated pipeline patterns to convert into parameterized_subdag usage. Files: `docs/hamilton_best_in_class_subdag_candidates.md`, `src/codeintel/build/hamilton/native/**`

### Phase 1 - Schema and Tag Metadata Foundation
- [ ] Sweep all table-producing nodes to apply `schema_output(...)` and canonical dataset tags. Files: `src/codeintel/sdk/annotations.py`, `src/codeintel/build/hamilton/native/**`, `src/codeintel/build/hamilton/tagging.py`
- [ ] Ensure materialize nodes emit domain/target/table_key/output_kind and materialization tags. Files: `src/codeintel/build/hamilton/save_to.py`, `src/codeintel/build/hamilton/materializers/**`, `src/codeintel/build/hamilton/tag_spec.py`

### Phase 2 - Output Validation with Data Quality Modifiers
- [ ] Apply warn/fail validators to key outputs (row count, nullability, PK uniqueness). Files: `src/codeintel/build/hamilton/data_quality.py`, `src/codeintel/build/hamilton/native/**`, `src/codeintel/sdk/validation.py`
- [ ] Tag validation nodes for UI hiding and ensure validation mode config is honored. Files: `src/codeintel/build/hamilton/tagging.py`, `src/codeintel/build/hamilton/validate.py`, `src/codeintel/runtime/compose.py`

### Phase 3 - Materialization Tags and Catalog Alignment
- [ ] Confirm `materialization`/`materialized_name` tags are propagated from saver nodes and visible in the semantic catalog. Files: `src/codeintel/build/hamilton/save_to.py`, `src/codeintel/serving/semantic/registry_compiler.py`
- [ ] Align saver tag schemas with UI expectations for table/artifact outputs. Files: `src/codeintel/build/hamilton/materializers/**`, `src/codeintel/serving/semantic/models.py`

### Phase 4 - Tag-Driven CLI and Runtime Querying
- [ ] Surface tag metadata in CLI outputs (targets listing/build results). Files: `src/codeintel/cli/handlers/targets.py`, `src/codeintel/cli/handlers/build.py`, `src/codeintel/cli/core/result_types.py`
- [ ] Add/update CLI snapshot tests to cover tag output rendering. Files: `tests/build/hamilton/snapshots/**`, `tests/build/hamilton/test_cli_snapshots.py`

### Phase 5 - Cross-Cutting Normalization with mutate
- [ ] Centralize ingestion normalization transforms and apply via mutate/pipe patterns. Files: `src/codeintel/build/hamilton/transforms/**`, `src/codeintel/build/hamilton/native/ingestion/**`
- [ ] Tag normalization steps for lineage visibility. Files: `src/codeintel/build/hamilton/tagging.py`

### Phase 6 - Parameterized SubDAGs and Config-Driven Composition
- [ ] Replace repeated pipeline patterns with parameterized_subdag/parameterize_sources. Files: `src/codeintel/build/hamilton/native/ingestion/**`, `src/codeintel/build/hamilton/native/graphs/**`, `src/codeintel/build/hamilton/nodes/support_nodes.py`
- [ ] Gate pipeline variants via config (resolve_from_config) with namespaced keys. Files: `src/codeintel/runtime/compose.py`, `src/codeintel/core/runtime/loader.py`

### Phase 7 - Dynamic Execution with Parallelizable/Collect
- [ ] Expand Parallelizable/Collect beyond module ingest where safe and add gating tests. Files: `src/codeintel/build/hamilton/native/**`, `tests/build/hamilton/**`
- [ ] Document dynamic execution knobs and defaults. Files: `docs/hamilton_best_in_class_implementation_plan.md`, `codeintel.yaml`

### Phase 8 - Adapter Expansion and ResultBuilder
- [ ] Add adapter safeguards (materialize-node locking/grouping) for parallel execution. Files: `src/codeintel/build/hamilton/adapters/parallel.py`, `src/codeintel/build/hamilton/executor.py`
- [ ] Add optional backend tests for ResultBuilder behavior and adapter detection. Files: `tests/build/hamilton/**`

### Phase 9 - Builder-Level Materializers
- [ ] Wire config schema and docs for materializer selection. Files: `src/codeintel/core/config/settings.py`, `src/codeintel/core/runtime/loader.py`, `src/codeintel/runtime/compose.py`, `codeintel.yaml`
- [ ] Add tests covering materializer config selection and DAG tagging. Files: `tests/build/hamilton/**`

### Phase 10 - Cache Lineage Export and Audit Pipelines
- [ ] Ensure cache log ingestion produces DuckDB rows tied to run_id/target outputs. Files: `src/codeintel/build/hamilton/executor.py`, `src/codeintel/observability/cache_log_ingest.py`
- [ ] Add CLI integration tests for cache log ingestion paths. Files: `tests/cli/**`, `tests/observability/**`

### Testing and Verification Additions
- [ ] Tag presence and schema_output derivation coverage. Files: `tests/build/hamilton/**`
- [ ] Validation warn/fail coverage and hidden validation nodes. Files: `tests/build/hamilton/**`
- [ ] Tag-driven CLI selection and metadata output tests. Files: `tests/build/hamilton/**`, `tests/cli/**`
- [ ] Dynamic execution gating and adapter optionality tests. Files: `tests/build/hamilton/**`
- [ ] Cache log ingestion tests (empty, populated, and failure modes). Files: `tests/observability/**`

## Testing and Verification

- Unit tests for tag presence and schema_output derivation.
- Validation tests covering warn/fail mode and tag-hidden nodes.
- CLI tests for tag-driven selection and output rendering.
- Dynamic execution POC tests with Parallelizable/Collect gating.
- Adapter tests for optional backends and ResultBuilder compatibility.
- Cache log ingestion tests covering empty/no-op and populated cases.

Primary test locations
- `tests/build/hamilton/**`
- `tests/observability/**`
- `tests/serving/semantic/**`
- `tests/cli/**` (if present, or add new CLI coverage)

## Rollout Order (Recommended)

1. Phase 0-2 (metadata + validation) with no execution behavior changes.
2. Phase 3-4 (catalog tags + CLI) to surface observability and usability.
3. Phase 5-6 (pipeline refactors) after validation coverage is stable.
4. Phase 7-8 (dynamic execution + adapters) behind config gates.
5. Phase 9-10 (materializers + cache lineage) once runtime is stable.

## Risks and Mitigations

- Dynamic execution complexity: gate behind config and keep default off.
- Adapter dependency drift: optional adapters must be import-guarded.
- Validation noise: start in warn mode, promote to fail once baselines exist.
- Tag inconsistency: enforce tag validation in build guardrails.

## Decision Points

- When to enable dynamic execution in production (after POC success).
- Which outputs must be materialized vs in-memory only.
- Whether to add Ray/Dask as optional dependencies in the core build.

## Deliverables Checklist

- Schema tag coverage for all table outputs.
- Validation decorators applied to critical outputs.
- Materialization tags emitted and indexed.
- Tag-driven CLI commands and runtime tag queries.
- Reusable pipeline subDAGs and config-driven variants.
- Dynamic execution POC with Parallelizable/Collect.
- Optional adapters with ResultBuilder.
- Config-driven materializers.
- Cache lineage export and ingestion.
