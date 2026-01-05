# Hamilton Inference-First Alignment Plan

## Intent

Make inferred schemas and settings the canonical truth derived from Hamilton outputs.
Every build run produces schema observations that drive runtime alignment, validation,
and dataset tuning. Static schema text remains optional and is treated as hints only.

## Scope

- Inference-driven TableSchema and Arrow schema generation.
- Schema observations persisted per build run (schema bytes + stats + provenance).
- Runtime always uses latest inferred schema from registry for alignment and serving.
- Hamilton annotations become soft hints merged into inference output.
- Drift becomes observable signals (diffs + alerts), not blocking gates.
- Streaming-first implementation using RecordBatchReader and scan_batches.

## Non-goals (Explicitly Deferred)

- Schema pinning or schema version gating.
- Inference sampling policies (full-stream inference is assumed for now).
- Hard blocking on drift or schema mismatch.

## Guiding Principles

- Inference is authoritative; metadata hints cannot override observed truth.
- Unification is alignment-only; it never becomes canonical schema.
- Streaming-first: avoid eager to_table() or full materialization in inference paths.
- Observability over enforcement: surface drift, do not stop pipelines.
- Minimal config surface; prefer defaults derived from observed stats.

## Target Architecture Overview

Build-time (authoritative):
Hamilton output stream -> SchemaObservation -> Registry (schema bytes + stats)

Runtime (authoritative):
Registry latest inferred schema -> Alignment -> Serving/ingest/validation

## Data Model Additions

### SchemaObservation (new)

A per-run record containing:
- table_key, repo, commit, target_name, observed_at
- schema_hash, schema_digest
- arrow_schema_ipc_b64 (serialized schema bytes)
- column_stats: null_count, distinct_count, min/max, avg_len
- dataset_stats: row_count, bytes, row_groups, batch_size
- derived_settings: dictionary_encode_columns, row_group_size, data_page_size
- drift_summary: missing/extra columns, type changes (for observability)

Suggested storage:
- New table: metadata.schema_observations (append-only)
- Keep registry linkage in metadata.table_schema_registry + metadata.schema_versions

## Implementation Plan (Phased)

### Phase 0: Definitions and storage surface

Goals:
- Define SchemaObservation payload and registry storage format.
- Establish metadata merging rules (observed wins, hints fill only missing).

Work:
- Add SchemaObservation model and helpers in
  `src/codeintel/build/schemas/inference_service.py` (or a new module).
- Add metadata schema_observations table and DDL in
  `src/codeintel/storage/metadata/ddl.py`.
- Add persistence helpers in
  `src/codeintel/storage/tracking/schema_catalog.py` and
  `src/codeintel/storage/tracking/schema_catalog_compile.py`.

Acceptance criteria:
- New table exists and can store observation records.
- Renderer cache continues to store arrow_schema_ipc_b64.

### Phase 1: Materializers emit schema observations (streaming)

Goals:
- Emit inference events directly from streaming outputs.

Work:
- In `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`,
  capture RecordBatchReader and emit SchemaObservation during write.
- In `src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py`,
  emit SchemaObservation from fetch_arrow_reader() output.
- Ensure observation capture never calls to_table().

Acceptance criteria:
- Observation records are emitted for both Arrow dataset and DuckDB outputs.
- No eager materialization required for inference.

### Phase 2: Inference service consumes batch iterators

Goals:
- Make inference fully streaming and stats-driven.

Work:
- Extend `src/codeintel/build/schemas/inference_service.py` to accept:
  - RecordBatchReader
  - Iterator[RecordBatch]
- Compute stats with pyarrow.compute (null_count, count_distinct, min/max, mean length).
- Derive TableSchema and Arrow schema directly from observed types.
- Merge Hamilton hints (nullable, description, PII) as soft defaults only.

Acceptance criteria:
- Inference runs without collecting full tables.
- TableSchema and Arrow schema reflect observed types plus hint metadata.

### Phase 3: Registry becomes inference-backed

Goals:
- Make inference outputs the canonical registry entries.

Work:
- Update `src/codeintel/core/schemas/service.py` to load inferred schema as default.
- Update `src/codeintel/storage/schema/arrow_schema.py` to prefer latest observation
  for schema bytes and metadata; remove static fallback paths.
- Ensure registry writes always include renderer_cache IPC bytes.

Acceptance criteria:
- Latest observation is the schema returned at runtime.
- Static schema sources no longer override inference output.

### Phase 4: Runtime alignment uses inferred schema everywhere

Goals:
- Eliminate runtime inference drift by always aligning to registry schema.

Work:
- `src/codeintel/serving/semantic/datasets.py`:
  pass inferred schema into dataset scanners and filter builders.
- `src/codeintel/serving/semantic/duckdb_relation_builder.py`:
  register relations with inferred schema where supported.
- `src/codeintel/storage/serving/snapshot_service.py`:
  load registry schema and pass to scanners on view creation.

Acceptance criteria:
- Serving and ingestion always align to the latest inferred schema.
- No schema inference occurs in serving paths.

### Phase 5: Inference-driven validation

Goals:
- Base validation constraints on observed data rather than static contracts.

Work:
- `src/codeintel/storage/validation/columnar.py`:
  use inferred nullability and type info from observations.
- `src/codeintel/build/exports/validation.py`:
  derive constraints from inference stats with JSON Schema fallback.

Acceptance criteria:
- Validation reflects inference outputs; drift is surfaced, not blocked.

### Phase 6: Inference-driven dataset tuning

Goals:
- Use observation stats to configure write/read performance.

Work:
- `src/codeintel/storage/datasets/arrow_store.py`:
  pick dictionary_encode/unify based on observed cardinality.
- `src/codeintel/storage/datasets/manifests.py`:
  persist inferred tuning knobs and stats.
- `src/codeintel/serving/semantic/engines/polars_engine.py`:
  enable set_sorted when inferred sort keys are present.

Acceptance criteria:
- Dataset writes use inferred row group sizing and dictionary settings.
- Serving benefits from sortedness metadata where safe.

### Phase 7: Drift observability (no gating)

Goals:
- Detect and surface drift for operators and developers.

Work:
- Add drift diff computation in inference output
  (missing/extra fields, type changes).
- Emit structured log metrics in build and serving pipelines.
- Add CLI/reporting in `src/codeintel/cli/handlers/meta.py` for drift summaries.

Acceptance criteria:
- Drift is visible in logs and reports without blocking execution.

## Plan Alignment (Graph-Authoritative + Advanced Hamilton)

Keep this plan in lockstep with
`docs/Make_Hamilton_graph_authoritative_partE.md` (see the executable checklist
in section "11.5 Executable Implementation Checklist"). Alignment points:

- SDK wrappers for `@schema.output`/`@tag`/`@check_output` become *hint-only*
  inputs to Phase 2 inference (no hard enforcement).
- Config-driven DAG shaping (`@config.when`/`@resolve`) must flow through
  `Builder.with_config(...)` so inference observations track the actual DAG
  instantiated for a run (Phase 1-2).
- Cache profile + JSONL logs supply run-level drift observability inputs
  referenced in Phase 7.
- Telemetry + semantic registry compiler use the same tag taxonomy that Phase 5
  validation depends on.
- Tag taxonomy enforcement in Hamilton validation complements Phase 5
  inference-driven validation.
- Optional dynamic execution requires Phase 2 streaming inference to operate on
  iterators (no eager materialization).

## Key Merge Rules (Inference vs Hints)

- Observed type always wins over hint type.
- Nullable is true if any nulls observed; otherwise hint may mark nullable true.
- Descriptions and PII tags are retained from hints when missing in observation.
- Extras policy defaults to "retain" when extras appear; otherwise "reject".

## Testing and Validation

- Unit tests for streaming inference on RecordBatchReader.
- Registry tests: renderer_cache contains inferred schema bytes.
- Serving tests: dataset scans use registry schema and align filters.
- Validation tests: inferred nullability and types produce expected results.
- Performance tests: no eager materialization in inference paths.

Suggested test files:
- `tests/build/schemas/test_inference_service.py`
- `tests/storage/schema/test_arrow_schema_registry.py`
- `tests/serving/semantic/test_dataset_alignment.py`
- `tests/storage/validation/test_columnar_inference_validation.py`
- `tests/validation/test_semantic_tag_taxonomy.py` (shared with PartE)
- `tests/semantic_registry/test_registry_compiler.py` (shared with PartE)
- `tests/observability/test_cache_log_ingest.py` (shared with PartE)
- `tests/runtime/test_dynamic_execution_profile.py` (shared with PartE)

## Rollout Strategy

1) Enable observation capture and registry writes (Phase 1-3).
2) Switch runtime to use inferred schemas everywhere (Phase 4).
3) Adopt inference-driven validation and tuning (Phase 5-6).
4) Add drift observability (Phase 7).

## Risks and Mitigations

- Schema churn: surface drift metrics and avoid blocking, rely on visibility.
- Type instability: prefer stable Arrow type mapping and unify only for alignment.
- Performance: keep inference streaming-only; avoid to_table().

## Open Questions

- Where to persist per-column stats (new table vs manifest extras)?
- Whether to expose inferred settings via runtime config overrides.
- How to encode extras policy thresholds without explicit sampling policy.
